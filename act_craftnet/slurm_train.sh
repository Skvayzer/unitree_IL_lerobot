#!/bin/bash
#SBATCH --job-name=act_craftnet
#SBATCH --partition=faculty
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=/vast/users/chenyuan.chen/constantine/act_craftnet/logs/slurm_%j.out
#SBATCH --error=/vast/users/chenyuan.chen/constantine/act_craftnet/logs/slurm_%j.err

set -e

# ── Environment ────────────────────────────────────────────────────────────────
source /vast/users/chenyuan.chen/miniconda3/etc/profile.d/conda.sh
conda activate unifolm_vla

# Prevent MIOpen read-only cache crash on shared filesystems
export MIOPEN_DISABLE_CACHE=1
# Prevent BF16 NaN on AMD MI210 (ROCm)
export HIPBLAS_OP_DTYPE_FP32=1
# NCCL / DDP
export NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_BLOCKING_WAIT=0

CODE_DIR="/vast/users/chenyuan.chen/constantine/act_craftnet"
DATA_DIR="/vast/users/chenyuan.chen/constantine/block_stacking"
OUT_DIR="/vast/users/chenyuan.chen/constantine/act_craftnet/runs"

mkdir -p "$OUT_DIR/act_craftnet_v3"
mkdir -p "$(dirname $SLURM_LOG_FILE 2>/dev/null || echo /tmp)"

echo "=== ACT-CraftNet v3 ==="
echo "  Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPUs: $SLURM_GPUS_ON_NODE"
echo "  Code: $CODE_DIR"
echo "  Data: $DATA_DIR"
echo "  Out:  $OUT_DIR"

cd "$CODE_DIR"

torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train.py \
    --data_dir="$DATA_DIR" \
    --output_dir="$OUT_DIR" \
    --run_name="act_craftnet_v3" \
    --chunk_size=50 \
    --dim_model=512 \
    --n_enc_layers=4 \
    --n_dec_layers=7 \
    --latent_dim=32 \
    --kl_weight=1.0 \
    --batch_size=8 \
    --steps=100000 \
    --lr=1e-4 \
    --warmup_steps=1000 \
    --weight_decay=1e-4 \
    --grad_clip=1.0 \
    --num_workers=4 \
    --skip_frames=1 \
    --val_frac=0.1 \
    --save_every=5000 \
    --log_every=100 \
    --wandb_project="act-craftnet"
