"""
BlockStackingDataset — reads the custom xr_teleoperate episode format.

State:   [left_arm(7) | left_ee(7) | right_arm(7) | right_ee(7)] = 28D
Action:  same 28D
Tactile: [left_ee(9)  | right_ee(9)] = 18D
Env:     [red_rel(3)  | yellow_rel(3) | green_rel(3)] = 9D (relative block xyz)
Cameras: color_0 (head), color_2 (left wrist), color_3 (right wrist)
Depths:  head_left_0 (head left), left_wrist_2, right_wrist_3   [3 views]

Each __getitem__ returns a window:
  obs_t  = frame[i]
  action_chunk = frames[i+1 : i+1+chunk_size]  (28D × chunk_size)
"""

import json
import os
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


CAMERA_KEYS    = ["color_0", "color_2", "color_3"]   # head, left wrist, right wrist
DEPTH_KEYS     = ["head_left_0", "left_wrist_2", "right_wrist_3"]  # 3 depth views
BLOCK_KEYS     = ["red_block", "yellow_block", "green_block"]
IMG_SIZE       = (224, 224)
DEPTH_IMG_SIZE = (120, 160)   # H×W: half of 240×320 stored size

# ImageNet normalisation for RGB cameras
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_image(path: str) -> np.ndarray:
    """Load JPEG/PNG as (3, H, W) float32 in [0,1], ImageNet-normalised."""
    img = cv2.imread(path)
    if img is None:
        return np.zeros((3, *IMG_SIZE), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)   # (3, H, W)


def _load_depth(path: str) -> np.ndarray:
    """Load uint16 PNG depth (mm) → float32 metres, resized to (1, H, W)."""
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)   # MUST use IMREAD_UNCHANGED for uint16
    if d is None or d.dtype != np.uint16:
        return np.zeros((1, *DEPTH_IMG_SIZE), dtype=np.float32)
    d = d.astype(np.float32) / 1000.0            # mm → metres
    d = cv2.resize(d, (DEPTH_IMG_SIZE[1], DEPTH_IMG_SIZE[0]),
                   interpolation=cv2.INTER_NEAREST)
    return d[np.newaxis]                          # (1, H, W)


def _state_from_frame(frame: dict) -> np.ndarray:
    """Extract 28D state vector from a frame dict."""
    states = frame["states"]
    parts = []
    for key in ["left_arm", "left_ee", "right_arm", "right_ee"]:
        qpos = states.get(key, {}).get("qpos", [])
        parts.extend(qpos if qpos else [0.0] * 7)
    return np.array(parts, dtype=np.float32)   # (28,)


def _action_from_frame(frame: dict) -> np.ndarray:
    """Extract 28D action vector from a frame dict."""
    actions = frame["actions"]
    parts = []
    for key in ["left_arm", "left_ee", "right_arm", "right_ee"]:
        qpos = actions.get(key, {}).get("qpos", [])
        parts.extend(qpos if qpos else [0.0] * 7)
    return np.array(parts, dtype=np.float32)   # (28,)


def _tactile_from_frame(frame: dict) -> np.ndarray:
    """Extract 18D tactile vector [left_ee(9) | right_ee(9)]."""
    tac = frame.get("tactiles", {})
    left  = tac.get("left_ee",  [0.0] * 9) or [0.0] * 9
    right = tac.get("right_ee", [0.0] * 9) or [0.0] * 9
    return np.array(left + right, dtype=np.float32)   # (18,)


def _env_from_frame(frame: dict) -> np.ndarray:
    """Extract 9D env state: relative xyz of red/yellow/green blocks."""
    op = (frame.get("sim_state") or {}).get("object_positions", {})
    parts = []
    for bk in BLOCK_KEYS:
        rel = (op.get(bk) or {}).get("rel", [0.0, 0.0, 0.0])
        parts.extend(rel[:3] if len(rel) >= 3 else [0.0, 0.0, 0.0])
    return np.array(parts, dtype=np.float32)   # (9,)


class EpisodeIndex:
    """Lightweight index over all episodes + frame offsets."""
    def __init__(self, dataset_dir: str, min_frames: int = 10):
        self.dataset_dir = Path(dataset_dir)
        self.episodes = []   # list of (ep_dir, [frame_dicts])
        for ep_dir in sorted(self.dataset_dir.iterdir()):
            if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
                continue
            json_path = ep_dir / "data.json"
            if not json_path.exists():
                continue
            with open(json_path) as f:
                raw = json.load(f)
            frames = raw.get("data", [])
            if len(frames) >= min_frames:
                self.episodes.append((ep_dir, frames))

    def __len__(self):
        return len(self.episodes)


class BlockStackingDataset(Dataset):
    """
    Each item: observation at time t + action chunk [t+1 … t+chunk_size].
    Images are (n_cam, 3, 224, 224), depths are (n_depth, 1, 120, 160).
    """
    def __init__(
        self,
        dataset_dir: str,
        chunk_size: int = 50,
        skip_frames: int = 1,
        augment: bool = False,
        episode_list: Optional[list] = None,
    ):
        self.chunk_size  = chunk_size
        self.skip_frames = skip_frames
        self.augment     = augment

        idx = EpisodeIndex(dataset_dir)
        episodes = idx.episodes if episode_list is None else [idx.episodes[i] for i in episode_list]

        self.samples = []   # (ep_dir, frame_dicts, obs_idx)
        for ep_dir, frames in episodes:
            for i in range(0, len(frames) - 1, skip_frames):
                self.samples.append((ep_dir, frames, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        ep_dir, frames, obs_idx = self.samples[idx]

        obs_frame = frames[obs_idx]

        # ── Images (n_cam, 3, 224, 224) ─────────────────────────────────────
        images = []
        for ck in CAMERA_KEYS:
            rel = (obs_frame.get("colors") or {}).get(ck)
            path = str(ep_dir / rel) if rel else ""
            images.append(_load_image(path))
        images = np.stack(images, axis=0)   # (3, 3, 224, 224)

        # ── Depths (n_depth, 1, 120, 160) ───────────────────────────────────
        depths = []
        for dk in DEPTH_KEYS:
            rel = (obs_frame.get("depths") or {}).get(dk)
            path = str(ep_dir / rel) if rel else ""
            depths.append(_load_depth(path))
        depths = np.stack(depths, axis=0)   # (3, 1, 120, 160)

        # ── Low-dim state ────────────────────────────────────────────────────
        state   = _state_from_frame(obs_frame)    # (28,)
        tactile = _tactile_from_frame(obs_frame)  # (18,)
        env_st  = _env_from_frame(obs_frame)      # (9,)

        # ── Action chunk ─────────────────────────────────────────────────────
        action_list = []
        action_is_pad = []
        for k in range(1, self.chunk_size + 1):
            fi = obs_idx + k
            if fi < len(frames):
                action_list.append(_action_from_frame(frames[fi]))
                action_is_pad.append(False)
            else:
                action_list.append(np.zeros(28, dtype=np.float32))
                action_is_pad.append(True)
        action       = np.stack(action_list, axis=0)      # (chunk, 28)
        action_is_pad = np.array(action_is_pad, dtype=bool)  # (chunk,)

        # ── Optional augmentation ─────────────────────────────────────────────
        if self.augment:
            # Random brightness/contrast jitter per camera
            for ci in range(len(CAMERA_KEYS)):
                if random.random() < 0.5:
                    alpha = random.uniform(0.8, 1.2)   # contrast
                    images[ci] = np.clip(images[ci] * alpha, -3, 3)

        return {
            "images":       torch.from_numpy(images),             # (3, 3, 224, 224)
            "depths":       torch.from_numpy(depths),             # (3, 1, 120, 160)
            "state":        torch.from_numpy(state),              # (28,)
            "tactile":      torch.from_numpy(tactile),            # (18,)
            "env_state":    torch.from_numpy(env_st),             # (9,)
            "action":       torch.from_numpy(action),             # (chunk, 28)
            "action_is_pad":torch.from_numpy(action_is_pad),     # (chunk,)
        }


def make_splits(
    dataset_dir: str,
    val_frac: float = 0.1,
    chunk_size: int = 50,
    skip_frames: int = 1,
    augment: bool = True,
) -> tuple:
    """Split episodes into train/val, return two BlockStackingDataset objects."""
    idx = EpisodeIndex(dataset_dir)
    n   = len(idx)
    n_val = max(1, int(n * val_frac))
    all_ep = list(range(n))
    random.shuffle(all_ep)
    val_ep   = all_ep[:n_val]
    train_ep = all_ep[n_val:]

    train_ds = BlockStackingDataset(dataset_dir, chunk_size, skip_frames,
                                    augment=augment, episode_list=train_ep)
    val_ds   = BlockStackingDataset(dataset_dir, chunk_size, skip_frames,
                                    augment=False,   episode_list=val_ep)
    return train_ds, val_ds
