#!/bin/bash
#SBATCH --job-name=sd3-sample
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=logs/sampling/slurm_sd3_%j.log
#SBATCH --error=logs/sampling/slurm_sd3_%j.log
#SBATCH --partition=killable
#SBATCH --nodelist=n-501,n-502,n-503,n-601,n-602,n-801,n-802,n-803,n-804,n-805,n-806,rack-bgw-dgx1,rack-gww-dgx1

set -euo pipefail

PROJECT_DIR="/home/ML_courses/03683533_2025/or_tal_almog/almog/revised-annealing-guidance"
cd "$PROJECT_DIR"

mkdir -p logs/sampling

# Keep ALL caches/tmp inside this repo
TMP_ROOT="$PROJECT_DIR/tmp"
mkdir -p "$TMP_ROOT"

# Temp dirs
export TMPDIR="$TMP_ROOT/tmpdir"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$TMPDIR"

# XDG base cache
export XDG_CACHE_HOME="$TMP_ROOT/xdg_cache"
mkdir -p "$XDG_CACHE_HOME"

# Hugging Face caches
export HF_HOME="$TMP_ROOT/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export DIFFUSERS_CACHE="$HF_HOME/diffusers"
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$DIFFUSERS_CACHE"

# PyTorch / extensions caches
export TORCH_HOME="$TMP_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$TMP_ROOT/torch_extensions"
mkdir -p "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR"

# Triton / CUDA compilation caches (if used)
export TRITON_CACHE_DIR="$TMP_ROOT/triton"
export CUDA_CACHE_PATH="$TMP_ROOT/nv_cache"
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

# Matplotlib (avoid writing to $HOME)
export MPLCONFIGDIR="$TMP_ROOT/matplotlib"
mkdir -p "$MPLCONFIGDIR"

# pip cache (avoid ~/.cache/pip)
export PIP_CACHE_DIR="$TMP_ROOT/pip_cache"
mkdir -p "$PIP_CACHE_DIR"

# W&B: keep data in project dir and disable service mode (its Unix-socket
# transport uses /tmp which is unreliable on SLURM nodes).
export WANDB_DIR="$TMP_ROOT/wandb"
export WANDB_CACHE_DIR="$TMP_ROOT/wandb_cache"
export WANDB_DATA_DIR="$TMP_ROOT/wandb_data"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
export WANDB__REQUIRE_SERVICE=false

# Make sure logs are streamed to SLURM output
export PYTHONUNBUFFERED=1

# Load Hugging Face token from .env
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
    import torch  # noqa: F401
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

ensure_requirements() {
	"$PY" - <<'PY'
try:
    import diffusers  # noqa: F401
    import transformers  # noqa: F401
    import omegaconf  # noqa: F401
    import dotenv  # noqa: F401
    import sentencepiece  # noqa: F401
    import google.protobuf  # noqa: F401
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
		echo "Installing dependencies into venv..."

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

NGPUS="${SLURM_GPUS_ON_NODE:-4}"

echo "Python: $($PY -V)"
"$PY" - <<'PY'
import torch, sys
print('torch =', torch.__version__)
print('cuda available =', torch.cuda.is_available())
if not torch.cuda.is_available():
    print("FATAL: CUDA not available. Exiting to avoid hanging on CPU.", file=sys.stderr)
    sys.exit(1)
n = torch.cuda.device_count()
print(f'gpus = {n}')
for i in range(n):
    print(f'  gpu {i}: {torch.cuda.get_device_name(i)}')
PY

CKPT="${SD3_SAMPLE_CHECKPOINT:-}"
CKPT_ID="${SD3_SAMPLE_CHECKPOINT_ID:-}"
OUTPUT_ROOT="${SD3_SAMPLE_OUTPUT_ROOT:-results/final}/${SLURM_JOB_ID}_${CKPT_ID:-unknown}"

TORCHRUN="$ENV_DIR/bin/torchrun"
if [[ ! -x "$TORCHRUN" ]]; then
    TORCHRUN="$PY -m torch.distributed.run"
fi

if [[ -n "$CKPT" && -n "$CKPT_ID" ]]; then
    $TORCHRUN --nproc_per_node="$NGPUS" --standalone \
        scripts/batch_sample_sd3.py \
        --checkpoint "$CKPT" --checkpoint_id "$CKPT_ID" \
        --output_root "$OUTPUT_ROOT" \
        --baselines --force \
        "$@"

    # Generate analysis plots (single GPU, fast)
    W_PLOT_DIR="$OUTPUT_ROOT/$CKPT_ID"

    echo "Generating w_scale_analysis plot..."
    "$PY" -u scripts/plot_w_scale_analysis.py \
        --checkpoint "$CKPT" \
        --output "$W_PLOT_DIR/w_scale_analysis.png" \
        --lr_label "$CKPT_ID"

    echo "Generating w heatmap..."
    "$PY" -u scripts/plot_w_heatmap.py \
        --checkpoint "$CKPT" \
        --output "$W_PLOT_DIR/w_heatmap.png" \
        --lr_label "$CKPT_ID" \
        --lambdas 0.0 0.4 0.6 0.8 1.0

    echo "Generating interpretability plots..."
    "$PY" -u scripts/plot_interpretability.py \
        --checkpoint "$CKPT" \
        --output_dir "$W_PLOT_DIR" \
        --lr_label "$CKPT_ID"

    # --- Fig2 comparison (woman in black dress + two dogs) ---
    echo "Generating fig2 comparison..."
    FIG2_DIR="$OUTPUT_ROOT/$CKPT_ID/fig2"
    "$PY" -u scripts/fig2_comparison.py \
        --checkpoint "$CKPT" \
        --output_dir "$FIG2_DIR"

    # --- FID / CLIP / ImageReward evaluation (COCO 5k) ---
    COCO_DIR="$PROJECT_DIR/data/coco2017"
    if [[ ! -f "$COCO_DIR/annotations/captions_val2017.json" ]]; then
        echo "Downloading COCO 2017 val..."
        PYTHON_BIN="$PY" bash scripts/download_coco.sh "$COCO_DIR"
    fi
    # Install metric deps (idempotent)
    "$PY" -m pip install -q clean-fid open_clip_torch image-reward 2>/dev/null || true

    echo "Running FID/CLIP/ImageReward evaluation..."
    $TORCHRUN --nproc_per_node="$NGPUS" --standalone \
        scripts/eval_metrics.py \
        --checkpoint "$CKPT" \
        --output_dir "$OUTPUT_ROOT/$CKPT_ID" \
        --coco_dir "$COCO_DIR" \
        --label "$CKPT_ID"
else
    "$PY" -u scripts/sample_sd3.py
fi
