"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import time
import torch
import logging

import numpy as np
from pprint import pformat
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext

from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from multiprocessing.sharedctypes import SynchronizedArray

from unitree_lerobot.eval_robot.make_robot import (
    setup_image_client,
    setup_robot_interface,
    process_images_and_observations,
)
from unitree_lerobot.eval_robot.utils.utils import (
    cleanup_resources,
    predict_action,
    to_list,
    to_scalar,
    EvalRealConfig,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data
from unitree_lerobot.eval_robot.utils.dex3_order import reorder_dex3_right_legacy_sim

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


def eval_policy(
    cfg: EvalRealConfig,
    policy: torch.nn.Module,
    dataset: LeRobotDataset,
    preprocessor,
    postprocessor,
):
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    logger_mp.info(f"Arguments: {cfg}")

    if cfg.visualization:
        rerun_logger = RerunLogger()

    policy.reset()  # Set policy to evaluation mode

    image_info = None
    try:
        # --- Setup Phase ---
        image_info = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)

        # Unpack interfaces for convenience
        arm_ctrl, arm_ik, ee_ctrl, ee_shared_mem, arm_dof, ee_dof = (
            robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_ctrl", "ee_shared_mem", "arm_dof", "ee_dof"]
        )
        img_client, tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam = (
            image_info[key]
            for key in [
                "img_client",
                "tv_img_array",
                "wrist_img_array",
                "tv_img_shape",
                "wrist_img_shape",
                "is_binocular",
                "has_wrist_cam",
            ]
        )

        # Get initial pose from the first step of the selected episode
        first_episode_idx = dataset.episodes[0] if dataset.episodes else 0
        episode_metadata = dataset.meta.episodes[int(first_episode_idx)]
        from_idx = int(episode_metadata["dataset_from_index"])
        step = dataset[from_idx]
        init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()

        user_input = input("Enter 's' to initialize the robot and start the evaluation: ")
        idx = 0
        print(f"user_input: {user_input}")
        full_state = None
        dex3_legacy_order_shim = bool(cfg.ee and cfg.ee.lower() == "dex3" and cfg.dex3_right_order_legacy_sim)
        if dex3_legacy_order_shim:
            logger_mp.warning(
                "Dex3 legacy sim right-hand order shim is ENABLED. "
                "Applying reorder on policy input/output for right hand."
            )
        else:
            logger_mp.info("Dex3 legacy sim right-hand order shim is disabled.")
        if user_input.lower() == "s":
            # "The initial positions of the robot's arm and fingers take the initial positions during data recording."
            logger_mp.info("Initializing robot to starting pose...")
            tau = robot_interface["arm_ik"].solve_tau(init_arm_pose)
            robot_interface["arm_ctrl"].ctrl_dual_arm(init_arm_pose, tau)
            time.sleep(1.0)  # Give time for the robot to move
            # --- Run Main Loop ---
            logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz.")
            diag_enabled = bool(cfg.enable_io_latency_diag)
            diag_every_n_steps = max(1, int(cfg.io_latency_diag_every_n_steps))
            diag_last_report_time = time.perf_counter()
            diag_last_report_idx = 0
            last_step_timing = {
                "obs_ms": 0.0,
                "policy_ms": 0.0,
                "act_ms": 0.0,
                "sleep_ms": 0.0,
                "loop_ms": 0.0,
            }
            while True:
                loop_start_time = time.perf_counter()
                # 1. Get Observations
                obs_start_time = time.perf_counter()
                observation, current_arm_q = process_images_and_observations(
                    tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam, arm_ctrl
                )
                left_ee_state = right_ee_state = np.array([])
                if cfg.ee:
                    with ee_shared_mem["lock"]:
                        full_state = np.array(ee_shared_mem["state"][:])
                        left_ee_state = full_state[:ee_dof]
                        right_ee_state = full_state[ee_dof:]
                    if dex3_legacy_order_shim:
                        right_ee_state = reorder_dex3_right_legacy_sim(
                            right_ee_state, context="eval_g1/right_ee_state"
                        )
                state_tensor = torch.from_numpy(
                    np.concatenate((current_arm_q, left_ee_state, right_ee_state), axis=0)
                ).float()
                observation["observation.state"] = state_tensor
                obs_end_time = time.perf_counter()
                # 2. Get Action from Policy
                policy_start_time = time.perf_counter()
                action = predict_action(
                    observation,
                    policy,
                    get_safe_torch_device(policy.config.device),
                    policy.config.use_amp,
                    step["task"],
                    use_dataset=cfg.use_dataset,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                action_np = action.cpu().numpy()
                policy_end_time = time.perf_counter()
                # 3. Execute Action
                act_start_time = time.perf_counter()
                arm_action = action_np[:arm_dof]
                tau = arm_ik.solve_tau(arm_action)
                arm_ctrl.ctrl_dual_arm(arm_action, tau)

                if cfg.ee:
                    ee_action_start_idx = arm_dof
                    left_ee_action = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
                    right_ee_action = action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof]
                    if dex3_legacy_order_shim:
                        right_ee_action = reorder_dex3_right_legacy_sim(
                            right_ee_action, context="eval_g1/right_ee_action"
                        )
                    # logger_mp.info(f"EE Action: left {left_ee_action}, right {right_ee_action}")

                    if isinstance(ee_shared_mem["left"], SynchronizedArray):
                        ee_shared_mem["left"][:] = to_list(left_ee_action)
                        ee_shared_mem["right"][:] = to_list(right_ee_action)
                    elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                        ee_shared_mem["left"].value = to_scalar(left_ee_action)
                        ee_shared_mem["right"].value = to_scalar(right_ee_action)
                act_end_time = time.perf_counter()

                if cfg.visualization:
                    visualization_data(idx, observation, state_tensor.numpy(), action_np, rerun_logger)
                idx += 1
                # Maintain frequency
                sleep_time = max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time))
                time.sleep(sleep_time)
                loop_end_time = time.perf_counter()
                last_step_timing = {
                    "obs_ms": (obs_end_time - obs_start_time) * 1000.0,
                    "policy_ms": (policy_end_time - policy_start_time) * 1000.0,
                    "act_ms": (act_end_time - act_start_time) * 1000.0,
                    "sleep_ms": sleep_time * 1000.0,
                    "loop_ms": (loop_end_time - loop_start_time) * 1000.0,
                }
                if diag_enabled and (idx - diag_last_report_idx) >= diag_every_n_steps:
                    now = time.perf_counter()
                    steps = max(1, idx - diag_last_report_idx)
                    dt = max(1e-6, now - diag_last_report_time)
                    loop_hz = float(steps) / dt
                    diag_last_report_idx = idx
                    diag_last_report_time = now

                    arm_state_age_ms = (
                        float(arm_ctrl.get_lowstate_age_ms()) if hasattr(arm_ctrl, "get_lowstate_age_ms") else float("nan")
                    )
                    ee_state_age_ms = (
                        float(ee_ctrl.get_state_age_ms()) if ee_ctrl is not None and hasattr(ee_ctrl, "get_state_age_ms") else float("nan")
                    )
                    image_stats = img_client.get_runtime_stats() if img_client is not None else {}
                    image_age_s = image_stats.get("last_frame_age_s", None)
                    image_latency_s = image_stats.get("avg_latency_s", None)
                    image_fps = image_stats.get("recv_fps_1s", float("nan"))
                    image_age_ms = image_age_s * 1000.0 if image_age_s is not None else float("nan")
                    image_latency_ms = image_latency_s * 1000.0 if image_latency_s is not None else None

                    if image_latency_ms is None:
                        image_latency_text = "N/A (no sender timestamp header)"
                    else:
                        image_latency_text = f"{image_latency_ms:.1f}"

                    policy_stats = (
                        policy.get_dual_rate_runtime_stats()
                        if hasattr(policy, "get_dual_rate_runtime_stats")
                        else {}
                    )
                    logger_mp.info(
                        "[I/O DIAG] loop_hz=%.2f obs_ms=%.1f policy_ms=%.1f act_ms=%.1f sleep_ms=%.1f "
                        "image_fps=%.2f image_age_ms=%.1f image_latency_ms=%s "
                        "arm_lowstate_age_ms=%.1f ee_state_age_ms=%.1f "
                        "s1_fallback=%s s1_queue=%s s2_busy=%s s2_ready=%s s2_refresh_age_ms=%s",
                        loop_hz,
                        last_step_timing["obs_ms"],
                        last_step_timing["policy_ms"],
                        last_step_timing["act_ms"],
                        last_step_timing["sleep_ms"],
                        image_fps,
                        image_age_ms,
                        image_latency_text,
                        arm_state_age_ms,
                        ee_state_age_ms,
                        policy_stats.get("s1_fallback_count", "n/a"),
                        policy_stats.get("s1_queue_len", "n/a"),
                        policy_stats.get("s2_worker_busy", "n/a"),
                        policy_stats.get("s2_chunks_ready", "n/a"),
                        (
                            f"{float(policy_stats.get('s2_last_refresh_age_ms', float('nan'))):.1f}"
                            if "s2_last_refresh_age_ms" in policy_stats
                            else "n/a"
                        ),
                    )
    except Exception as e:
        logger_mp.info(f"An error occurred: {e}")
    finally:
        if image_info:
            cleanup_resources(image_info)


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")

    dataset = LeRobotDataset(repo_id=cfg.repo_id)

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(cfg.policy, dataset_stats=dataset.meta.stats)
    policy.eval()

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(
            cfg=cfg,
            policy=policy,
            dataset=dataset,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
