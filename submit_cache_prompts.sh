#!/bin/bash
#SBATCH --job-name=cache-sd3-prompts
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=logs/ddp/cache/slurm_cache_prompts_%j.log
#SBATCH --error=logs/ddp/cache/slurm_cache_prompts_%j.log
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students

set -euo pipefail

PROJECT_DIR="/home/ML_courses/03683533_2025/or_tal_almog/almog/revised-annealing-guidance"
cd "$PROJECT_DIR"

mkdir -p logs/ddp

TMP_ROOT="$PROJECT_DIR/tmp"
mkdir -p "$TMP_ROOT"

export TMPDIR="$TMP_ROOT/tmpdir"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$TMPDIR"

export XDG_CACHE_HOME="$TMP_ROOT/xdg_cache"
mkdir -p "$XDG_CACHE_HOME"

export HF_HOME="$TMP_ROOT/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export DIFFUSERS_CACHE="$HF_HOME/diffusers"
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$DIFFUSERS_CACHE"

export TORCH_HOME="$TMP_ROOT/torch"
mkdir -p "$TORCH_HOME"

export PIP_CACHE_DIR="$TMP_ROOT/pip_cache"
mkdir -p "$PIP_CACHE_DIR"

export PYTHONUNBUFFERED=1

# Load Hugging Face token from .env
if [[ -f "$PROJECT_DIR/.env" ]]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | grep -E 'HUGGINGFACE_HUB_TOKEN|HF_TOKEN' | xargs)
fi

ENV_DIR="${ENV_DIR:-$PROJECT_DIR/venv}"
PY="$ENV_DIR/bin/python"

echo "Python: $($PY -V)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
echo "Starting prompt caching..."

CACHE_DIR="${CACHE_DIR:-/home/ML_courses/03683533_2025/or_tal_almog/almog/prompt_cache}"
IMAGE_ROOT="${IMAGE_ROOT:-src/data/laion/laion_pop_images}"

NPROC=8
"$PY" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port=$((29500 + RANDOM % 1000)) \
    scripts/cache_prompts_sd3.py \
    --image_root "$IMAGE_ROOT" \
    --cache_dir "$CACHE_DIR" \
    --batch_size 256
