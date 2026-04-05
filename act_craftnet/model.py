"""
ACT-CraftNet model v2 (bimanual G1+Dex3, 28D state/action).

Architecture:
  Visual encoder:  FrozenDINOv2 (ViT-B/14, shared) → avg-pool patch tokens → (B, 512) per cam
  Depth encoder:   DepthCNNEncoder: depth → 2D conv pyramid → global pool → (B, 256D)
  System 0:        MoE reactive finger correction with bidirectional S1↔S0 connections
  Encoder tokens:  [z | state | tactile_fb | env | depth | cam0 | cam1 | cam2] = 8 tokens
  VAE encoder:     [CLS | state | action_chunk] → μ, log_σ → latent z (32D)
  Decoder:         chunk_size learned queries cross-attend over encoder output
  Action head:     Linear(512, 28)

Bidirectional connections:
  S1→S0: physical_intent = Linear(latent_dim, 128) applied to CVAE latent z
  S0→S1: tactile_feedback = FeedbackEncoder(tactile, finger_state) → 64D
          injected as extra encoder token (near-zero init, no circular dependency)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

class ACTConfig:
    state_dim:    int = 28
    action_dim:   int = 28
    tactile_dim:  int = 18
    env_dim:      int = 9
    n_cameras:    int = 3      # RGB cameras
    n_depth_views: int = 3    # depth cameras (head + 2 wrists)
    finger_dim:   int = 14    # left_ee(7) + right_ee(7)
    chunk_size:   int = 50
    latent_dim:   int = 32
    dim_model:    int = 512
    n_heads:      int = 8
    dim_ff:       int = 3200
    n_enc_layers: int = 4
    n_dec_layers: int = 7
    dropout:      float = 0.1
    kl_weight:    float = 1.0
    physical_intent_dim: int = 128   # S1→S0 channel dim
    feedback_dim: int = 64           # S0→S1 tactile feedback dim
    depth_h:      int = 120
    depth_w:      int = 160

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sinusoidal_embed(seq_len: int, dim: int, device) -> torch.Tensor:
    """Returns (1, seq_len, dim) sinusoidal positional encoding."""
    pos = torch.arange(seq_len, device=device).unsqueeze(1).float()
    i   = torch.arange(0, dim, 2, device=device).float()
    div = torch.exp(i * (-math.log(10000.0) / dim))
    pe  = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)   # (1, seq, dim)


class _TransformerEncoderLayer(nn.Module):
    def __init__(self, d: int, h: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=False)
        self.ff   = nn.Sequential(
            nn.Linear(d, d_ff), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(d_ff, d)
        )
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.dp1, self.dp2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        a, _ = self.attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.n1(x + self.dp1(a))
        x = self.n2(x + self.dp2(self.ff(x)))
        return x


class _TransformerDecoderLayer(nn.Module):
    def __init__(self, d: int, h: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=False)
        self.cross_attn = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=False)
        self.ff = nn.Sequential(
            nn.Linear(d, d_ff), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(d_ff, d)
        )
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.n3 = nn.LayerNorm(d)
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)
        self.dp3 = nn.Dropout(dropout)

    def forward(self, tgt, mem, mem_key_padding_mask=None):
        a, _ = self.self_attn(tgt, tgt, tgt)
        tgt = self.n1(tgt + self.dp1(a))
        a, _ = self.cross_attn(tgt, mem, mem, key_padding_mask=mem_key_padding_mask)
        tgt = self.n2(tgt + self.dp2(a))
        tgt = self.n3(tgt + self.dp3(self.ff(tgt)))
        return tgt


# ──────────────────────────────────────────────────────────────────────────────
# FrozenDINOv2 visual encoder
# ──────────────────────────────────────────────────────────────────────────────

class FrozenDINOv2(nn.Module):
    """
    DINOv2 ViT-B/14 frozen visual backbone.

    ViT-B/14: patch_size=14, image=224 → grid 16×16 = 256 patch tokens, hidden_dim=768.
    We avg-pool the 256 patch tokens and project 768→out_dim.
    All DINOv2 parameters are frozen (no gradient, not in optimizer).
    """
    def __init__(self, out_dim: int = 512):
        super().__init__()
        from transformers import Dinov2Model
        self.dino = Dinov2Model.from_pretrained("facebook/dinov2-base")
        for p in self.dino.parameters():
            p.requires_grad_(False)
        self.proj = nn.Linear(768, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) → (B, out_dim)"""
        with torch.no_grad():
            out = self.dino(pixel_values=x)
        # last_hidden_state: (B, 257, 768) — CLS + 256 patch tokens
        patch_tokens = out.last_hidden_state[:, 1:]   # (B, 256, 768)
        pooled = patch_tokens.mean(dim=1)              # (B, 768)
        return self.proj(pooled)                       # (B, out_dim)


# ──────────────────────────────────────────────────────────────────────────────
# iDP3 depth encoder
# ──────────────────────────────────────────────────────────────────────────────

class iDP3DepthEncoder(nn.Module):
    """
    iDP3-style depth encoder (Ze et al. 2024): depth image → 3D point cloud → Conv1d pyramid → token.

    Pipeline per depth view:
      1. depth (1, H, W) metres → back-project to 3D using real camera intrinsics
      2. Filter valid pixels (depth_min < d < depth_max)
      3. Random subsample to n_points (512) valid points, zero-pad if fewer
      4. Subtract centroid of valid points only (translation invariance)
      5. Conv1d(3→64)+BN+ReLU → global max pool → 64D
         Conv1d(64→128)+BN+ReLU → global max pool → 128D
         Conv1d(128→256)+BN+ReLU → global max pool → 256D
         Multi-scale concat → 448D per view
    All views concatenated (n_views × 448D) → fusion MLP(1344→512→256) → out_dim token.

    Camera intrinsics at 120×160 (loaded resolution):
      Derived by scaling factory calibration (640×480) × 0.25:
        head:   fx=fy=138.5, cx=80.0, cy=60.0  (Isaac Sim ~60° hFOV)
        wrists: fx=fy=151.25, cx=80.0, cy=60.0  (D405 factory calibration)
    """
    # Per-view intrinsics at 120×160: (fx, fy, cx, cy)
    # view 0 = head camera, view 1 = left wrist, view 2 = right wrist
    DEFAULT_INTRINSICS = {
        0: (138.5,  138.5,  80.0, 60.0),   # head (Isaac Sim, scaled 640→120/480→160)
        1: (151.25, 151.25, 80.0, 60.0),   # left wrist (D405, scaled)
        2: (151.25, 151.25, 80.0, 60.0),   # right wrist (D405, scaled)
    }

    def __init__(self, n_views: int = 3, depth_h: int = 120, depth_w: int = 160,
                 n_points: int = 512, out_dim: int = 256,
                 depth_min: float = 0.02, depth_max: float = 2.0,
                 intrinsics: dict | None = None):
        super().__init__()
        self.n_views   = n_views
        self.depth_h   = depth_h
        self.depth_w   = depth_w
        self.n_points  = n_points
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.intrinsics = intrinsics or self.DEFAULT_INTRINSICS

        # Shared Conv1d pyramid (point-wise, kernel=1 along point dimension)
        self.conv1 = nn.Conv1d(3,   64,  1, bias=False)
        self.conv2 = nn.Conv1d(64,  128, 1, bias=False)
        self.conv3 = nn.Conv1d(128, 256, 1, bias=False)
        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(128)
        self.bn3   = nn.BatchNorm1d(256)

        # Cross-view fusion MLP: n_views × 448D → out_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(n_views * 448, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
            nn.LayerNorm(out_dim),
        )

        # Pre-computed pixel-coordinate grids per view (saved as buffers)
        for v in range(n_views):
            fx, fy, cx, cy = self.intrinsics[v]
            us = torch.arange(depth_w).float()
            vs = torch.arange(depth_h).float()
            grid_v, grid_u = torch.meshgrid(vs, us, indexing="ij")  # (H, W)
            # Normalised directions: (u - cx) / fx and (v - cy) / fy
            dir_x = (grid_u - cx) / fx   # (H, W)
            dir_y = (grid_v - cy) / fy   # (H, W)
            ones  = torch.ones_like(dir_x)
            # Stack as (3, H*W): [dir_x_flat, dir_y_flat, ones_flat]
            dirs  = torch.stack([dir_x, dir_y, ones], dim=0).reshape(3, -1)  # (3, N_pix)
            self.register_buffer(f"dirs_{v}", dirs)

    def _to_point_cloud(self, depth: torch.Tensor, view_idx: int) -> torch.Tensor:
        """
        Back-project depth image to 3D using real camera intrinsics.

        Args:
            depth:    (B, 1, H, W) float32 metres
            view_idx: 0=head, 1=left_wrist, 2=right_wrist

        Returns:
            (B, 3, n_points) point cloud, centroid-subtracted.
            Invalid / missing depth regions are zero-padded.
        """
        B, _, H, W = depth.shape
        device     = depth.device
        N_pix      = H * W

        dirs = getattr(self, f"dirs_{view_idx}")   # (3, N_pix) — pre-computed

        d_flat = depth.reshape(B, N_pix)           # (B, N_pix)

        # Valid pixel mask
        valid = (d_flat >= self.depth_min) & (d_flat <= self.depth_max)  # (B, N_pix)

        # 3D points: dirs * depth  →  (B, 3, N_pix)
        pts_all = dirs.unsqueeze(0) * d_flat.unsqueeze(1)  # (B, 3, N_pix)

        results = []
        for b in range(B):
            valid_b = valid[b]                     # (N_pix,)
            pts_b   = pts_all[b, :, valid_b]       # (3, N_valid)
            n_valid = pts_b.shape[1]

            if n_valid == 0:
                results.append(torch.zeros(3, self.n_points, device=device))
                continue

            # Random subsample to n_points
            if n_valid >= self.n_points:
                idx = torch.randperm(n_valid, device=device)[:self.n_points]
                pts_b = pts_b[:, idx]              # (3, n_points)
            else:
                pad = torch.zeros(3, self.n_points - n_valid, device=device)
                pts_b = torch.cat([pts_b, pad], dim=1)

            # Centroid subtraction over valid points only (not over padding)
            centroid = pts_b[:, :n_valid].mean(dim=1, keepdim=True)
            pts_b    = pts_b - centroid

            results.append(pts_b)

        return torch.stack(results, dim=0)         # (B, 3, n_points)

    def _encode_view(self, pts: torch.Tensor) -> torch.Tensor:
        """pts: (B, 3, n_points) → multi-scale feature (B, 448)."""
        x1 = F.relu(self.bn1(self.conv1(pts)))    # (B, 64, n_points)
        f1 = x1.max(dim=2).values                 # (B, 64)
        x2 = F.relu(self.bn2(self.conv2(x1)))    # (B, 128, n_points)
        f2 = x2.max(dim=2).values                 # (B, 128)
        x3 = F.relu(self.bn3(self.conv3(x2)))    # (B, 256, n_points)
        f3 = x3.max(dim=2).values                 # (B, 256)
        return torch.cat([f1, f2, f3], dim=1)     # (B, 448)

    def forward(self, depths: torch.Tensor) -> torch.Tensor:
        """
        depths: (B, n_views, 1, H, W) float32 metres
                view order: 0=head, 1=left_wrist, 2=right_wrist
        Returns: (B, out_dim)
        """
        view_feats = []
        for v in range(self.n_views):
            pts  = self._to_point_cloud(depths[:, v], v)  # (B, 3, n_points)
            feat = self._encode_view(pts)                  # (B, 448)
            view_feats.append(feat)
        multi_view = torch.cat(view_feats, dim=1)          # (B, n_views*448)
        return self.fusion_mlp(multi_view)                 # (B, out_dim)


# ──────────────────────────────────────────────────────────────────────────────
# CNN depth encoder (drop-in replacement for iDP3DepthEncoder)
# ──────────────────────────────────────────────────────────────────────────────

class DepthCNNEncoder(nn.Module):
    """
    Simple CNN depth encoder — no intrinsics, no point cloud.

    Treats each depth map as a 1-channel image. Extracts multi-scale
    spatial features via 2D convolutions, global-pools each scale,
    concatenates across views, and projects to 256D.

    Input:  (B, n_views, 1, H, W)  float32 depth in metres
    Output: (B, 256)
    """

    def __init__(
        self,
        n_views: int = 3,
        out_dim: int = 256,
        depth_max: float = 2.0,
    ):
        super().__init__()
        self.n_views   = n_views
        self.depth_max = depth_max

        # 2D conv pyramid — same channel progression as iDP3 (3→64→128→256)
        # but 2D instead of 1D, and over pixels instead of points.
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Fusion: n_views × 448 → out_dim
        self.fusion = nn.Sequential(
            nn.Linear(n_views * 448, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, depths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depths: (B, n_views, 1, H, W) float32 depth in metres
        Returns:
            (B, out_dim) fused depth feature
        """
        view_feats = []
        for v in range(self.n_views):
            d = depths[:, v]                              # (B, 1, H, W)
            d = d.clamp(0, self.depth_max) / self.depth_max  # normalise [0,1]

            x1 = self.conv1(d)                            # (B,  64, H/2, W/2)
            x2 = self.conv2(x1)                           # (B, 128, H/4, W/4)
            x3 = self.conv3(x2)                           # (B, 256, H/8, W/8)

            f1 = self.gap(x1).flatten(1)                  # (B,  64)
            f2 = self.gap(x2).flatten(1)                  # (B, 128)
            f3 = self.gap(x3).flatten(1)                  # (B, 256)

            view_feats.append(torch.cat([f1, f2, f3], dim=-1))  # (B, 448)

        fused = torch.cat(view_feats, dim=-1)             # (B, n_views*448)
        return self.fusion(fused)                         # (B, out_dim)


# ──────────────────────────────────────────────────────────────────────────────
# System 0: MoE reactive finger correction
# ──────────────────────────────────────────────────────────────────────────────

class System0Policy(nn.Module):
    """
    System 0: Mixture-of-Experts reactive finger correction.

    Bidirectional connections with ACT (System 1):
      S1→S0: physical_intent (128D) derived from enc_out[0] (post-attention latent
             token) — meaningful at both train and inference, unlike raw z which
             is zeros at inference.
      S0→S1: tactile_feedback (64D) injected as extra ACT encoder token.

    MoE: 4 experts, top-k=2, soft routing via gumbel-free softmax.
    Expert input: cat([enc_out[0](512) | physical_intent(128) | tactile(18) |
                       finger_state(14) | s1_fingers(14)]) = 686D
      - enc_out[0], physical_intent, tactile, finger_state: global context (B, D)
        expanded to (B, T, D) for per-timestep processing
      - s1_fingers: S1's planned finger targets per timestep (B, T, 14)
        → S0 learns "given what S1 planned at step t, compute residual correction"
    Output: delta_finger (B, T, 14) — time-varying, one correction per timestep.
    Near-zero init ensures the system starts as a pass-through (no correction).

    Two-phase usage in ACTCraftNet.forward:
      Phase 1 (before encoder): encode_feedback(tactile, finger_state) → tactile_feedback token
      Phase 2 (after decoder):  forward_delta(enc_out[0], physical_intent, tactile,
                                              finger_state, s1_fingers) → delta_finger (B,T,14)

    Finger layout (28D state/action):
      left_ee:  dims  7-13
      right_ee: dims 21-27
      finger_state = state[:, 7:14] + state[:, 21:28] concatenated → 14D
    """
    N_EXPERTS     = 4
    TOP_K         = 2
    FINGER_DIM    = 14    # left_ee(7) + right_ee(7)
    EXPERT_HIDDEN = 256

    def __init__(self, hidden_dim: int = 512, physical_intent_dim: int = 128,
                 tactile_dim: int = 18, feedback_dim: int = 64):
        super().__init__()
        D  = hidden_dim
        PI = physical_intent_dim
        # Expert/router input: enc_out[0] + physical_intent + tactile + finger_state + s1_fingers
        expert_in = D + PI + tactile_dim + self.FINGER_DIM + self.FINGER_DIM  # 512+128+18+14+14=686

        # Router: conditioned on full expert context
        self.router = nn.Linear(expert_in, self.N_EXPERTS)

        # Experts: full context → Δfinger (tactile in input → reactive to contact)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_in, self.EXPERT_HIDDEN),
                nn.ReLU(inplace=True),
                nn.Linear(self.EXPERT_HIDDEN, self.FINGER_DIM),
            )
            for _ in range(self.N_EXPERTS)
        ])

        # Feedback encoder: [tactile(18) | finger_state(14)] → feedback_dim
        # Called in Phase 1 (before encoder) — provides S0→S1 token
        self.feedback_encoder = nn.Sequential(
            nn.Linear(tactile_dim + self.FINGER_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feedback_dim),
        )

        self._init_near_zero()

    def _init_near_zero(self):
        """Near-zero init so S0 starts as identity (safe at training start)."""
        gain = 0.01
        for expert in self.experts:
            nn.init.normal_(expert[-1].weight, std=gain)
            nn.init.zeros_(expert[-1].bias)
        nn.init.normal_(self.feedback_encoder[-1].weight, std=gain)
        nn.init.zeros_(self.feedback_encoder[-1].bias)

    def encode_feedback(
        self,
        tactile: torch.Tensor,       # (B, 18)
        finger_state: torch.Tensor,  # (B, 14)
    ) -> torch.Tensor:
        """
        Phase 1 — called BEFORE the main encoder.
        Produces the S0→S1 tactile feedback token.
        Returns: tactile_feedback (B, feedback_dim)
        """
        fb_in = torch.cat([tactile, finger_state], dim=-1)   # (B, 32)
        return self.feedback_encoder(fb_in)                   # (B, feedback_dim)

    def forward_delta(
        self,
        hidden_state: torch.Tensor,    # (B, 512) — enc_out[0], post-attention latent
        physical_intent: torch.Tensor, # (B, 128) — intent_proj(enc_out[0])
        tactile: torch.Tensor,         # (B, 18)
        finger_state: torch.Tensor,    # (B, 14) — current finger positions
        s1_fingers: torch.Tensor,      # (B, T, 14) — S1's planned finger targets
    ) -> torch.Tensor:
        """
        Phase 2 — called AFTER the decoder (actions_hat available).
        Computes per-timestep residual finger correction.
        Returns: delta_finger (B, T, 14)

        Global context (hidden_state, physical_intent, tactile, finger_state) is
        expanded across T timesteps and concatenated with s1_fingers so each expert
        computes "given S1 planned X at step t, what correction is needed?"
        """
        T = s1_fingers.shape[1]

        # Expand global context to (B, T, D) for per-timestep processing
        h  = hidden_state.unsqueeze(1).expand(-1, T, -1)     # (B, T, 512)
        pi = physical_intent.unsqueeze(1).expand(-1, T, -1)  # (B, T, 128)
        ta = tactile.unsqueeze(1).expand(-1, T, -1)          # (B, T, 18)
        fs = finger_state.unsqueeze(1).expand(-1, T, -1)     # (B, T, 14)

        # Full context: 512+128+18+14+14 = 686D per timestep
        ctx = torch.cat([h, pi, ta, fs, s1_fingers], dim=-1)  # (B, T, 686)

        logits              = self.router(ctx)                           # (B, T, N_EXPERTS)
        topk_vals, topk_idx = logits.topk(self.TOP_K, dim=-1)           # (B, T, TOP_K)
        gates               = F.softmax(topk_vals, dim=-1)              # (B, T, TOP_K)

        # All expert outputs: run each expert on full (B*T, 686) context
        BT = s1_fingers.shape[0] * T
        ctx_flat = ctx.reshape(BT, -1)                                   # (B*T, 686)
        all_out = torch.stack(
            [exp(ctx_flat).reshape(s1_fingers.shape[0], T, self.FINGER_DIM)
             for exp in self.experts], dim=2
        )                                                                # (B, T, N, 14)

        idx_exp      = topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.FINGER_DIM)  # (B,T,K,14)
        selected     = all_out.gather(2, idx_exp)                        # (B, T, TOP_K, 14)
        delta_finger = (selected * gates.unsqueeze(-1)).sum(2)           # (B, T, 14)

        return delta_finger


# ──────────────────────────────────────────────────────────────────────────────
# ACT-CraftNet (main policy)
# ──────────────────────────────────────────────────────────────────────────────

class ACTCraftNet(nn.Module):
    def __init__(self, cfg: ACTConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.dim_model

        # ── Visual backbone: FrozenDINOv2 (shared across cameras) ────────────
        self.dino_cam  = FrozenDINOv2(out_dim=D)   # frozen, not in optimizer

        # ── Depth encoder: CNN (3 depth views → 1 token) ────────────────────
        self.depth_encoder = DepthCNNEncoder(
            n_views=cfg.n_depth_views,
            out_dim=256,
            depth_max=2.0,
        )
        self.depth_proj = nn.Linear(256, D)

        # ── System 0: MoE finger correction ──────────────────────────────────
        self.system0 = System0Policy(
            hidden_dim=D,
            physical_intent_dim=cfg.physical_intent_dim,
            tactile_dim=cfg.tactile_dim,
            feedback_dim=cfg.feedback_dim,
        )

        # ── S1→S0 physical intent projection ─────────────────────────────────
        # Source: enc_out[0] (post-attention latent token, 512D) — not raw z.
        # enc_out[0] is contextualized by all other tokens even when z=zeros,
        # so it carries meaningful intent signal at both train and inference.
        self.physical_intent_proj = nn.Linear(cfg.dim_model, cfg.physical_intent_dim)

        # ── S0→S1 feedback projection (near-zero init) ────────────────────────
        self.tactile_feedback_proj = nn.Linear(cfg.feedback_dim, D)
        nn.init.normal_(self.tactile_feedback_proj.weight, std=0.01)
        nn.init.zeros_(self.tactile_feedback_proj.bias)

        # ── Low-dim projections ───────────────────────────────────────────────
        self.state_proj   = nn.Linear(cfg.state_dim,   D)
        self.env_proj     = nn.Linear(cfg.env_dim,     D)
        self.latent_proj  = nn.Linear(cfg.latent_dim,  D)

        # ── VAE encoder (CVAE — sees state + action chunk) ────────────────────
        n_vae_tokens = 1 + 1 + cfg.chunk_size   # [CLS | state | actions…]
        self.vae_cls_embed   = nn.Embedding(1, D)
        self.vae_state_proj  = nn.Linear(cfg.state_dim,  D)
        self.vae_action_proj = nn.Linear(cfg.action_dim, D)
        self.register_buffer(
            "vae_pos_enc",
            _sinusoidal_embed(n_vae_tokens, D, torch.device("cpu")).squeeze(0)
        )
        vae_enc_layer = nn.TransformerEncoderLayer(
            D, cfg.n_heads, cfg.dim_ff, cfg.dropout,
            batch_first=False, norm_first=False
        )
        self.vae_encoder     = nn.TransformerEncoder(vae_enc_layer, num_layers=4)
        self.vae_latent_proj = nn.Linear(D, cfg.latent_dim * 2)

        # ── Main transformer encoder ───────────────────────────────────────────
        # Tokens: [z | state | tactile_fb | env | depth | cam0 | cam1 | cam2] = 8
        n_enc_tokens = 5 + cfg.n_cameras   # 5 non-camera + 3 cameras = 8
        self.register_buffer(
            "enc_pos_enc",
            _sinusoidal_embed(n_enc_tokens, D, torch.device("cpu")).squeeze(0)
        )
        enc_layer = nn.TransformerEncoderLayer(
            D, cfg.n_heads, cfg.dim_ff, cfg.dropout,
            batch_first=False, norm_first=False
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_enc_layers)

        # ── Main transformer decoder ───────────────────────────────────────────
        self.query_embed  = nn.Embedding(cfg.chunk_size, D)
        self.decoder      = nn.ModuleList([
            _TransformerDecoderLayer(D, cfg.n_heads, cfg.dim_ff, cfg.dropout)
            for _ in range(cfg.n_dec_layers)
        ])
        self.decoder_norm = nn.LayerNorm(D)

        # ── Action head ────────────────────────────────────────────────────────
        self.action_head = nn.Linear(D, cfg.action_dim)   # (B, chunk, 28)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            # Skip frozen DINOv2 and near-zero initialised params
            if "dino_cam.dino" in name:
                continue
            if "system0" in name or "tactile_feedback_proj" in name:
                continue   # already initialised in System0Policy
            if p.dim() > 1 and not isinstance(p, nn.Embedding):
                nn.init.xavier_uniform_(p)

    def _encode_vae(self, state, action_chunk, action_is_pad):
        """Encode (state, action_chunk) → (μ, log_σ, z)."""
        B  = state.shape[0]
        D  = self.cfg.dim_model
        device = state.device

        cls  = self.vae_cls_embed.weight.unsqueeze(0).expand(B, 1, -1)  # (B,1,D)
        st   = self.vae_state_proj(state).unsqueeze(1)                   # (B,1,D)
        acts = self.vae_action_proj(action_chunk)                         # (B,chunk,D)
        seq  = torch.cat([cls, st, acts], dim=1)                          # (B,2+chunk,D)

        pos  = self.vae_pos_enc.unsqueeze(0)                              # (1,2+chunk,D)
        seq  = seq + pos

        cls_state_pad = torch.zeros(B, 2, dtype=torch.bool, device=device)
        pad_mask = torch.cat([cls_state_pad, action_is_pad], dim=1)

        out = self.vae_encoder(seq.permute(1, 0, 2),
                               src_key_padding_mask=pad_mask)[0]          # (B, D) CLS

        params    = self.vae_latent_proj(out)
        mu        = params[:, :self.cfg.latent_dim]
        log_sigma = params[:, self.cfg.latent_dim:]
        z = mu + log_sigma.div(2).exp() * torch.randn_like(mu)
        return mu, log_sigma, z

    def _extract_finger_state(self, state: torch.Tensor) -> torch.Tensor:
        """Extract 14D finger state from 28D state: left_ee[7:14] + right_ee[21:28]."""
        return torch.cat([state[:, 7:14], state[:, 21:28]], dim=-1)  # (B, 14)

    def forward(self, batch: dict) -> tuple:
        """
        Training forward.
        Returns: (actions_hat, mu, log_sigma)
          actions_hat: (B, chunk, 28)  — already includes S0 finger correction
        """
        device = batch["state"].device
        B      = batch["state"].shape[0]
        D      = self.cfg.dim_model
        cfg    = self.cfg

        state     = batch["state"].float()         # (B, 28)
        tactile   = batch["tactile"].float()       # (B, 18)
        env_state = batch["env_state"].float()     # (B, 9)
        images    = batch["images"].float()        # (B, n_cam, 3, 224, 224)
        depths    = batch["depths"].float()        # (B, n_depth, 1, 120, 160)

        # ── CVAE encode ────────────────────────────────────────────────────────
        if self.training and "action" in batch:
            mu, log_sigma, z = self._encode_vae(
                state, batch["action"].float(), batch["action_is_pad"]
            )
        else:
            mu = log_sigma = None
            z  = torch.zeros(B, cfg.latent_dim, device=device)

        # ── S0 Phase 1: tactile feedback token (before encoder) ─────────────────
        # Feedback encoder only — does not need enc_out yet.
        finger_state     = self._extract_finger_state(state)        # (B, 14)
        tactile_feedback = self.system0.encode_feedback(tactile, finger_state)  # (B, 64)
        tac_fb_token     = self.tactile_feedback_proj(tactile_feedback)         # (B, D)

        # ── DINOv2 camera features ──────────────────────────────────────────────
        # Process all cameras: flatten B×cam, run DINOv2, reshape
        B_cam = B * cfg.n_cameras
        imgs_flat = images.reshape(B_cam, 3, 224, 224)          # (B*3, 3, 224, 224)
        cam_feats_flat = self.dino_cam(imgs_flat)                # (B*3, D)
        cam_feats = cam_feats_flat.reshape(B, cfg.n_cameras, D) # (B, 3, D)

        # ── iDP3 depth token ───────────────────────────────────────────────────
        depth_feat = self.depth_encoder(depths)   # (B, 256)
        depth_tok  = self.depth_proj(depth_feat)  # (B, D)

        # ── Build encoder token sequence (seq_first) ───────────────────────────
        # Order: [z | state | tactile_fb | env | depth | cam0 | cam1 | cam2]
        tokens = [
            self.latent_proj(z),          # (B, D)
            self.state_proj(state),       # (B, D)
            tac_fb_token,                 # (B, D)  S0→S1
            self.env_proj(env_state),     # (B, D)
            depth_tok,                    # (B, D)  iDP3
        ]
        for ci in range(cfg.n_cameras):
            tokens.append(cam_feats[:, ci])   # (B, D)

        seq = torch.stack(tokens, dim=0)         # (8, B, D)
        pos = self.enc_pos_enc.unsqueeze(1)      # (8, 1, D)
        seq = seq + pos

        mem = self.encoder(seq)                  # (8, B, D)

        # ── S0 Phase 2a: physical intent from enc_out[0] ─────────────────────────
        physical_intent = self.physical_intent_proj(mem[0])          # (B, 128)

        # ── Transformer decoder ────────────────────────────────────────────────
        queries = self.query_embed.weight.unsqueeze(1).expand(-1, B, -1)  # (chunk, B, D)
        out = queries
        for layer in self.decoder:
            out = layer(out, mem)
        out = self.decoder_norm(out)             # (chunk, B, D)
        out = out.permute(1, 0, 2)              # (B, chunk, D)

        actions_hat = self.action_head(out)      # (B, chunk, 28)

        # ── S0 Phase 2b: per-timestep residual finger correction ───────────────
        # s1_fingers: S1's planned finger targets, (B, chunk, 14)
        # forward_delta runs after decoder so actions_hat is available.
        s1_fingers = torch.cat([
            actions_hat[:, :, 7:14],    # (B, chunk, 7) left_ee
            actions_hat[:, :, 21:28],   # (B, chunk, 7) right_ee
        ], dim=-1)                                                     # (B, chunk, 14)

        delta_finger = self.system0.forward_delta(
            mem[0], physical_intent, tactile, finger_state, s1_fingers
        )                                                              # (B, chunk, 14)

        # Add time-varying delta residually (detach s1_fingers from delta path
        # is NOT needed — gradients should flow through both branches)
        actions_hat = actions_hat.clone()   # avoid in-place on graph leaf
        actions_hat[:, :, 7:14]  = actions_hat[:, :, 7:14]  + delta_finger[:, :, :7]
        actions_hat[:, :, 21:28] = actions_hat[:, :, 21:28] + delta_finger[:, :, 7:]

        return actions_hat, mu, log_sigma

    @torch.no_grad()
    def predict(self, batch: dict) -> torch.Tensor:
        """Inference: returns (B, chunk, 28) action chunk."""
        self.eval()
        actions_hat, _, _ = self.forward(batch)
        return actions_hat


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

def compute_loss(actions_hat, mu, log_sigma, batch, kl_weight: float = 1.0):
    """L1 action loss (including S0-corrected finger channels) + KL divergence."""
    gt_action     = batch["action"].float()    # (B, chunk, 28)
    action_is_pad = batch["action_is_pad"]     # (B, chunk) bool

    mask = (~action_is_pad).unsqueeze(-1)      # (B, chunk, 1)

    l1 = (F.l1_loss(actions_hat, gt_action, reduction="none") * mask).sum() / mask.sum()

    kl = torch.tensor(0.0, device=actions_hat.device)
    if mu is not None and log_sigma is not None:
        kl = (-0.5 * (1 + log_sigma - mu.pow(2) - log_sigma.exp())).sum(-1).mean()

    loss = l1 + kl_weight * kl
    return loss, {
        "l1_loss":    l1.item(),
        "kl_loss":    kl.item(),
        "total_loss": loss.item(),
    }
