"""CraftNet remote policy server.

Loads the policy on GPU and serves action predictions over ZMQ.
Runs on the remote PC with the GPU that can fit the model.

Usage:
    cd unitree_IL_lerobot
    python -m unitree_lerobot.eval_robot.policy_server \
        --checkpoint_path /path/to/checkpoint/pretrained_model \
        --repo_id unitreerobotics/G1_Dex3_BlockStacking_Dataset \
        --host 0.0.0.0 --port 5556 --device cuda:0
"""

import argparse
import io
import time

import numpy as np
import torch
import zmq
import msgpack
import msgpack_numpy as m
from PIL import Image

m.patch()

from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.utils import get_safe_torch_device

# Trigger processor step registration for all policy types
import lerobot.policies.groot_n16.processor_groot_n16  # noqa: F401


def decompress_image_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    buf = io.BytesIO(jpeg_bytes)
    img = Image.open(buf)
    return np.array(img)


class PolicyServer:
    """ZMQ REQ/REP policy inference server.

    Protocol:
    - Client sends msgpack dict:
        {"type": "reset"|"predict"|"ping",
         "images": {cam_key: jpeg_bytes, ...},
         "state": [float, ...],
         "task": str}
    - Server replies msgpack dict:
        {"status": "ok"|"error", "action": [float, ...]|null}
    """

    def __init__(self, checkpoint_path, repo_id, device="cuda:0",
                 host="0.0.0.0", port=5556, rename_map=None):
        self.device = get_safe_torch_device(device)
        self.host = host
        self.port = port

        print(f"[PolicyServer] Loading dataset metadata from {repo_id}...")
        dataset = LeRobotDataset(repo_id=repo_id)
        dataset_stats = dataset.meta.stats
        if rename_map:
            dataset_stats = rename_stats(dataset_stats, rename_map)

        print(f"[PolicyServer] Loading policy from {checkpoint_path}...")
        from lerobot.configs.policies import PreTrainedConfig

        policy_config = PreTrainedConfig.from_pretrained(checkpoint_path)
        policy_config.pretrained_path = checkpoint_path

        self.policy = make_policy(
            cfg=policy_config,
            ds_meta=dataset.meta,
            rename_map=rename_map,
        )
        self.policy.to(self.device)
        self.policy.eval()

        preprocessor_overrides = {
            "device_processor": {"device": self.device.type},
            "normalizer_processor": {
                "stats": dataset_stats,
                "features": {
                    **self.policy.config.input_features,
                    **self.policy.config.output_features,
                },
                "norm_map": self.policy.config.normalization_mapping,
            },
        }
        if rename_map:
            preprocessor_overrides["rename_observations_processor"] = {
                "rename_map": rename_map,
            }
        postprocessor_overrides = {
            "unnormalizer_processor": {
                "stats": dataset_stats,
                "features": self.policy.config.output_features,
                "norm_map": self.policy.config.normalization_mapping,
            },
        }

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_config,
            pretrained_path=checkpoint_path,
            dataset_stats=dataset_stats,
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides=postprocessor_overrides,
        )

        print(f"[PolicyServer] Policy loaded on {self.device}")
        print(f"[PolicyServer] Type: {type(self.policy).__name__}")

    def predict(self, observation: dict, task: str) -> np.ndarray:
        """Run one prediction: preprocess -> policy -> postprocess."""
        from copy import copy
        from contextlib import nullcontext

        obs = copy(observation)
        obs["task"] = task

        with torch.inference_mode():
            if self.preprocessor is not None:
                processed = {}
                for name, value in obs.items():
                    tensor = (
                        torch.from_numpy(value)
                        if isinstance(value, np.ndarray)
                        else value
                    )
                    if isinstance(tensor, torch.Tensor) and "images" in name:
                        if tensor.dtype != torch.float32:
                            tensor = tensor.to(dtype=torch.float32)
                        tensor = tensor / 255.0
                        # HWC -> CHW
                        if (
                            tensor.ndim == 3
                            and tensor.shape[0] not in (1, 3, 4)
                            and tensor.shape[-1] in (1, 3, 4)
                        ):
                            tensor = tensor.permute(2, 0, 1).contiguous()
                    processed[name] = tensor
                processed["task"] = task
                policy_input = self.preprocessor(processed)
            else:
                policy_input = obs

            action = self.policy.select_action(policy_input)

            if self.postprocessor is not None:
                action = self.postprocessor(action)

            action = action.squeeze(0).cpu().numpy()

        return action

    def serve(self):
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REP)
        socket.bind(f"tcp://{self.host}:{self.port}")

        print(f"[PolicyServer] Listening on tcp://{self.host}:{self.port}")
        print(f"[PolicyServer] Waiting for client...")

        step = 0
        while True:
            try:
                msg_bytes = socket.recv()
                request = msgpack.unpackb(msg_bytes, raw=False)
                req_type = request.get("type", "predict")

                if req_type == "reset":
                    self.policy.reset()
                    if self.preprocessor is not None:
                        self.preprocessor.reset()
                    if self.postprocessor is not None:
                        self.postprocessor.reset()
                    step = 0
                    socket.send(
                        msgpack.packb({"status": "ok", "action": None},
                                      use_bin_type=True)
                    )
                    print("[PolicyServer] Reset")
                    continue

                if req_type == "ping":
                    socket.send(
                        msgpack.packb({"status": "ok", "action": None},
                                      use_bin_type=True)
                    )
                    continue

                t0 = time.perf_counter()

                # Decode images
                observation = {}
                for cam_key, jpeg_bytes in request.get("images", {}).items():
                    img_np = decompress_image_jpeg(jpeg_bytes)
                    observation[cam_key] = torch.from_numpy(img_np)

                # Decode state
                state = request.get("state", [])
                observation["observation.state"] = torch.tensor(
                    state, dtype=torch.float32
                )

                task = request.get("task", "")

                t1 = time.perf_counter()
                action = self.predict(observation, task)
                t2 = time.perf_counter()

                response = {"status": "ok", "action": action.tolist()}
                socket.send(msgpack.packb(response, use_bin_type=True))

                if step % 30 == 0:
                    print(
                        f"[PolicyServer] step {step} | "
                        f"decode: {(t1 - t0) * 1000:.1f}ms | "
                        f"predict: {(t2 - t1) * 1000:.1f}ms | "
                        f"action[:4]: {action[:4].round(3)}"
                    )
                step += 1

            except KeyboardInterrupt:
                print("\n[PolicyServer] Shutting down...")
                break
            except Exception as e:
                print(f"[PolicyServer] Error: {e}")
                import traceback
                traceback.print_exc()
                try:
                    socket.send(
                        msgpack.packb(
                            {"status": "error", "action": None, "error": str(e)},
                            use_bin_type=True,
                        )
                    )
                except Exception:
                    pass

        socket.close()
        ctx.term()


def main():
    parser = argparse.ArgumentParser(description="CraftNet Remote Policy Server")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--rename_map", type=str, default=None,
                        help="JSON string for camera rename map")
    args = parser.parse_args()

    rename_map = None
    if args.rename_map:
        import json
        rename_map = json.loads(args.rename_map)

    server = PolicyServer(
        checkpoint_path=args.checkpoint_path,
        repo_id=args.repo_id,
        device=args.device,
        host=args.host,
        port=args.port,
        rename_map=rename_map,
    )
    server.serve()


if __name__ == "__main__":
    main()
