#!/bin/bash
#SBATCH --job-name=eval-metrics
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sampling/slurm_eval_%j.log
#SBATCH --error=logs/sampling/slurm_eval_%j.log
#SBATCH --partition=killable
#SBATCH --nodelist=n-501,n-502,n-503,n-601,n-602,n-801,n-802,n-803,n-804,n-805,n-806,rack-bgw-dgx1,rack-gww-dgx1

set -euo pipefail

PROJECT_DIR="/home/ML_courses/03683533_2025/or_tal_almog/almog/revised-annealing-guidance"
cd "$PROJECT_DIR"

mkdir -p logs/sampling

# Keep ALL caches/tmp inside this repo
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
export TORCH_EXTENSIONS_DIR="$TMP_ROOT/torch_extensions"
mkdir -p "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR"

export TRITON_CACHE_DIR="$TMP_ROOT/triton"
export CUDA_CACHE_PATH="$TMP_ROOT/nv_cache"
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

export MPLCONFIGDIR="$TMP_ROOT/matplotlib"
mkdir -p "$MPLCONFIGDIR"

export PIP_CACHE_DIR="$TMP_ROOT/pip_cache"
mkdir -p "$PIP_CACHE_DIR"

export WANDB_DIR="$TMP_ROOT/wandb"
export WANDB_CACHE_DIR="$TMP_ROOT/wandb_cache"
export WANDB_DATA_DIR="$TMP_ROOT/wandb_data"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
export WANDB__REQUIRE_SERVICE=false

export PYTHONUNBUFFERED=1

if [[ -f "$PROJECT_DIR/.env" ]]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | grep -E 'HUGGINGFACE_HUB_TOKEN|HF_TOKEN' | xargs)
fi

DEPS_MARKER="$TMP_ROOT/deps_installed.ok"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_DIR="${ENV_DIR:-$PROJECT_DIR/venv}"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
	echo "No python found at $ENV_DIR/bin/python; creating a venv at $ENV_DIR using $PYTHON_BIN"
	"$PYTHON_BIN" -m venv "$ENV_DIR"
fi

PY="$ENV_DIR/bin/python"

ensure_torch() {
	"$PY" - <<'PY'
try:
    import torch
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

ensure_requirements() {
	"$PY" - <<'PY'
try:
    import diffusers, transformers, omegaconf, dotenv, sentencepiece, google.protobuf
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

if ensure_torch && ensure_requirements; then
	if [[ -f "$DEPS_MARKER" ]]; then
		echo "Deps marker found ($DEPS_MARKER); skipping pip installs."
	else
		touch "$DEPS_MARKER"
		echo "Deps already present; wrote marker ($DEPS_MARKER)."
	fi
else
	"$PY" -m pip install -q --upgrade pip wheel setuptools
	if ! ensure_torch; then
		TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
		TORCH_VERSION="${TORCH_VERSION:-2.3.1+cu121}"
		TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.18.1+cu121}"
		TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.3.1+cu121}"
		"$PY" -m pip install --upgrade --index-url "$TORCH_INDEX_URL" \
			"torch==${TORCH_VERSION}" \
			"torchvision==${TORCHVISION_VERSION}" \
			"torchaudio==${TORCHAUDIO_VERSION}"
	else
		echo "Installing requirements into venv..."
	fi
	"$PY" -m pip install -r requirements_slurm.txt
	touch "$DEPS_MARKER"
fi

# Install metric dependencies
"$PY" -m pip install -q clean-fid open_clip_torch image-reward 2>/dev/null || \
    "$PY" -m pip install -q clean-fid open-clip-torch image-reward 2>/dev/null || \
    echo "WARNING: Could not install some metric packages"

echo "Python: $($PY -V)"
"$PY" - <<'PY'
import torch, sys
print('torch =', torch.__version__)
print('cuda available =', torch.cuda.is_available())
if not torch.cuda.is_available():
    print("FATAL: CUDA not available.", file=sys.stderr); sys.exit(1)
n = torch.cuda.device_count()
print(f'gpus = {n}')
for i in range(n):
    print(f'  gpu {i}: {torch.cuda.get_device_name(i)}')
PY

# Download COCO if needed
COCO_DIR="$PROJECT_DIR/data/coco2017"
PYTHON_BIN="$PY" bash scripts/download_coco.sh "$COCO_DIR"

NGPUS="${SLURM_GPUS_ON_NODE:-4}"

CKPT="${EVAL_CHECKPOINT:-}"
LABEL="${EVAL_LABEL:-}"
OUTPUT_ROOT="${EVAL_OUTPUT_DIR:-results/eval/${SLURM_JOB_ID}_${LABEL:-unknown}}"

TORCHRUN="$ENV_DIR/bin/torchrun"
if [[ ! -x "$TORCHRUN" ]]; then
    TORCHRUN="$PY -m torch.distributed.run"
fi

BASELINE_INDICES="${EVAL_BASELINE_INDICES:-}"
SKIP_BASELINES="${EVAL_SKIP_BASELINES:-}"

EXTRA_ARGS=""
if [[ -n "$CKPT" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --checkpoint $CKPT"
fi
if [[ -n "$LABEL" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --label $LABEL"
fi
if [[ -n "$BASELINE_INDICES" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --baseline_indices $BASELINE_INDICES"
fi
if [[ -n "$SKIP_BASELINES" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --skip_baselines"
fi

echo "=== Starting FID/CLIP/ImageReward evaluation ==="
echo "Checkpoint: ${CKPT:-none (baselines only)}"
echo "Label: ${LABEL:-auto}"
echo "Output: $OUTPUT_ROOT"

$TORCHRUN --nproc_per_node="$NGPUS" --standalone \
    scripts/eval_metrics.py \
    --output_dir "$OUTPUT_ROOT" \
    --coco_dir "$COCO_DIR" \
    $EXTRA_ARGS
