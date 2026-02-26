"""Weights & Biases integration for annealing guidance training and sampling.

All W&B logic lives here. Existing scripts only need:
  - train_sd3.py:        import, init_training, log_train, finish  (4 lines)
  - batch_sample_sd3.py: import, init_sampling, finish             (3 lines)
"""
import os
import time
import torch

try:
    import wandb
except ImportError:
    wandb = None

_run = None
_mode = None
_guidance_model_ref = None
_train_guidance_data = []

# Time / progress tracking (training)
_train_start_time = None
_last_step_time = None
_max_steps = None
_loss_ema = None
_step_times = []  # rolling window for avg step time
_STEP_TIME_WINDOW = 50

# Time tracking (sampling)
_sample_start_time = None
_sample_image_count = 0
_sample_last_image_time = None


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

def login():
    if wandb is None:
        return
    key = os.getenv("WANDB_KEY")
    if key:
        wandb.login(key=key)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def init_training(config, guidance_model=None):
    """Start a W&B *training* run and register a forward hook on the guidance model."""
    global _run, _mode, _guidance_model_ref, _train_start_time, _max_steps
    global _last_step_time, _loss_ema, _step_times, _train_guidance_data
    if wandb is None:
        return None
    _mode = "train"
    _guidance_model_ref = guidance_model
    _train_start_time = time.time()
    _last_step_time = _train_start_time
    _max_steps = config["training"]["max_steps"]
    _loss_ema = None
    _step_times = []
    _train_guidance_data = []
    login()

    gpu_total_gb = None
    if torch.cuda.is_available():
        gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    _run = wandb.init(
        entity="annealing-guidance",
        project="annealing-guidance",
        job_type="train",
        config={
            "model_id": config["diffusion"]["model_id"],
            "max_steps": config["training"]["max_steps"],
            "batch_size": config["training"]["batch_size"],
            "lr": config["training"]["optimizer_kwargs"].get("lr"),
            "weight_decay": config["training"]["optimizer_kwargs"].get("weight_decay"),
            "accumulation_steps": config["training"].get("accumulation_steps", 1),
            "save_interval": config["training"]["save_interval"],
            "guidance_scale_model": config.get("guidance_scale_model", {}),
            "prompt_noise": config["training"].get("prompt_noise", {}),
            "seed": config.get("seed"),
            "low_memory": config.get("low_memory"),
        },
    )

    if guidance_model is not None:
        _register_train_hook(guidance_model)

    if torch.cuda.is_available():
        _run.config.update({
            "gpu": torch.cuda.get_device_name(0),
            "gpu_vram_gb": round(gpu_total_gb, 2),
        })
    return _run


def init_sampling(config_dict, guidance_model=None):
    """Start a W&B *sampling* run and hook the guidance model for trajectory capture."""
    global _run, _mode, _guidance_model_ref
    global _sample_start_time, _sample_image_count, _sample_last_image_time
    if wandb is None:
        return None
    _mode = "sample"
    _guidance_model_ref = guidance_model
    _sample_start_time = time.time()
    _sample_image_count = 0
    _sample_last_image_time = _sample_start_time
    login()

    _run = wandb.init(
        entity="annealing-guidance",
        project="annealing-guidance",
        job_type="sample",
        config=config_dict,
    )

    if guidance_model is not None:
        _register_sample_hook(guidance_model)

    if torch.cuda.is_available():
        _run.config.update({
            "gpu": torch.cuda.get_device_name(0),
            "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
        })
    return _run


# ---------------------------------------------------------------------------
# Forward hooks — capture guidance-scale predictions without touching callers
# ---------------------------------------------------------------------------

def _register_train_hook(model):
    """Store the last (timestep, w, lambda) on the model after each forward."""
    def hook(module, input, output):
        module._wb_last_w = output.detach()
        module._wb_last_t = input[0].detach() if isinstance(input[0], torch.Tensor) else torch.tensor(input[0])
        module._wb_last_lam = input[1].detach() if isinstance(input[1], torch.Tensor) else torch.tensor(input[1])
    model.register_forward_hook(hook)


def _register_sample_hook(model):
    """Accumulate every (timestep, w, lambda) prediction during sampling."""
    model._wb_sample_data = []
    model._wb_prev_t = None

    def hook(module, input, output):
        global _sample_image_count, _sample_last_image_time
        t = input[0]
        t_val = t.float().mean().item() if isinstance(t, torch.Tensor) else float(t)
        w_val = output.detach().float().mean().item()
        lam = input[1]
        lam_val = lam.float().mean().item() if isinstance(lam, torch.Tensor) else float(lam)
        module._wb_sample_data.append({"timestep": t_val, "guidance_scale": w_val, "lambda": lam_val})

        # Detect new generation start (timestep jumped back up) → log previous image timing
        if module._wb_prev_t is not None and t_val > module._wb_prev_t + 50:
            _sample_image_count += 1
            now = time.time()
            if _run is not None:
                elapsed = now - _sample_start_time
                img_time = now - _sample_last_image_time
                data = {
                    "sampling/images_generated": _sample_image_count,
                    "sampling/time_per_image_sec": round(img_time, 2),
                    "sampling/elapsed_min": round(elapsed / 60, 2),
                }
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    peak = torch.cuda.max_memory_allocated()
                    data["system/gpu_memory_gb"] = round(peak / (1024**3), 3)
                    data["system/gpu_utilization_pct"] = round(peak / props.total_memory * 100, 1)
                _run.log(data, step=_sample_image_count)
            _sample_last_image_time = now
        module._wb_prev_t = t_val
    model.register_forward_hook(hook)


# ---------------------------------------------------------------------------
# Per-step training logger
# ---------------------------------------------------------------------------

def log_train(step, loss, model):
    """Log loss, guidance-scale stats, time metrics, GPU utilization, and progress."""
    global _last_step_time, _loss_ema
    if _run is None:
        return

    now = time.time()

    # --- Loss metrics ---
    data = {"train/loss": loss}
    if _loss_ema is None:
        _loss_ema = loss
    else:
        _loss_ema = 0.97 * _loss_ema + 0.03 * loss
    data["train/loss_ema"] = _loss_ema

    # --- Guidance scale metrics (from forward hook) ---
    if hasattr(model, '_wb_last_w'):
        w = model._wb_last_w.float()
        data["train/guidance_scale_mean"] = w.mean().item()
        if w.numel() > 1:
            data["train/guidance_scale_min"] = w.min().item()
            data["train/guidance_scale_max"] = w.max().item()

    if hasattr(model, '_wb_last_t'):
        data["train/timestep_mean"] = model._wb_last_t.float().mean().item()

    if hasattr(model, '_wb_last_lam'):
        data["train/lambda_mean"] = model._wb_last_lam.float().mean().item()

    # --- GPU metrics ---
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        peak_alloc = torch.cuda.max_memory_allocated()
        data["system/gpu_memory_gb"] = round(peak_alloc / (1024**3), 3)
        data["system/gpu_utilization_pct"] = round(peak_alloc / props.total_memory * 100, 1)

    # --- Time metrics ---
    step_time = now - _last_step_time
    _step_times.append(step_time)
    if len(_step_times) > _STEP_TIME_WINDOW:
        _step_times.pop(0)
    avg_step_time = sum(_step_times) / len(_step_times)
    elapsed = now - _train_start_time

    data["time/step_time_sec"] = round(step_time, 4)
    data["time/avg_step_time_sec"] = round(avg_step_time, 4)
    data["time/steps_per_sec"] = round(1.0 / avg_step_time, 3) if avg_step_time > 0 else 0
    data["time/elapsed_min"] = round(elapsed / 60, 2)

    if _max_steps and step > 0:
        remaining_steps = max(_max_steps - step, 0)
        data["time/eta_min"] = round(remaining_steps * avg_step_time / 60, 1)

    # --- Progress ---
    if _max_steps:
        data["train/progress_pct"] = round(step / _max_steps * 100, 2)

    _last_step_time = now
    _run.log(data, step=step)

    # Accumulate for the final guidance-scale-vs-timestep scatter plot
    if hasattr(model, '_wb_last_w') and hasattr(model, '_wb_last_t'):
        for t_v, w_v in zip(
            model._wb_last_t.float().cpu().view(-1),
            model._wb_last_w.float().cpu().view(-1),
        ):
            _train_guidance_data.append([step, t_v.item(), w_v.item()])


# ---------------------------------------------------------------------------
# Finish — final charts + cleanup
# ---------------------------------------------------------------------------

def finish():
    """Log final guidance-scale charts, summary stats, and close the W&B run."""
    global _run, _mode, _guidance_model_ref, _train_guidance_data
    global _loss_ema, _step_times, _sample_image_count
    if _run is None:
        return
    try:
        if _mode == "train":
            _log_train_guidance_graph()
            _log_train_summary()
        elif _mode == "sample":
            _log_sample_guidance_graph()
            _log_sample_summary()
    finally:
        _run.finish()
        _run = None
        _mode = None
        _guidance_model_ref = None
        _train_guidance_data = []
        _loss_ema = None
        _step_times = []
        _sample_image_count = 0


def _log_train_summary():
    """Log summary metrics to the run (appear in the W&B runs table)."""
    elapsed = time.time() - _train_start_time if _train_start_time else 0
    summary = {
        "summary/total_time_min": round(elapsed / 60, 2),
    }
    if _step_times:
        summary["summary/avg_step_time_sec"] = round(sum(_step_times) / len(_step_times), 4)
    if _loss_ema is not None:
        summary["summary/final_loss_ema"] = _loss_ema
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        peak = torch.cuda.max_memory_allocated()
        summary["summary/peak_gpu_memory_gb"] = round(peak / (1024**3), 3)
        summary["summary/peak_gpu_utilization_pct"] = round(peak / props.total_memory * 100, 1)
    _run.summary.update(summary)


def _log_sample_summary():
    """Log summary metrics for the sampling run."""
    global _sample_image_count
    # Account for the final image (no subsequent hook call to trigger the count)
    if _guidance_model_ref and hasattr(_guidance_model_ref, '_wb_sample_data') and _guidance_model_ref._wb_sample_data:
        _sample_image_count += 1

    elapsed = time.time() - _sample_start_time if _sample_start_time else 0
    summary = {
        "summary/total_images": _sample_image_count,
        "summary/total_time_min": round(elapsed / 60, 2),
    }
    if _sample_image_count > 0:
        summary["summary/avg_time_per_image_sec"] = round(elapsed / _sample_image_count, 2)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        peak = torch.cuda.max_memory_allocated()
        summary["summary/peak_gpu_memory_gb"] = round(peak / (1024**3), 3)
        summary["summary/peak_gpu_utilization_pct"] = round(peak / props.total_memory * 100, 1)
    _run.summary.update(summary)


def _log_train_guidance_graph():
    """Scatter plot: predicted guidance scale vs. timestep across all training steps."""
    if not _train_guidance_data:
        return
    table = wandb.Table(data=_train_guidance_data, columns=["step", "timestep", "guidance_scale"])
    _run.log({
        "charts/guidance_scale_vs_timestep": wandb.plot.scatter(
            table, "timestep", "guidance_scale",
            title="Guidance Scale vs Timestep (Training)",
        ),
    })


def _log_sample_guidance_graph():
    """Line chart: guidance-scale trajectory per lambda during sampling."""
    model = _guidance_model_ref
    if model is None or not hasattr(model, '_wb_sample_data') or not model._wb_sample_data:
        return

    data = model._wb_sample_data

    # Split into per-generation trajectories (timestep jumps up → new generation)
    trajectories = []
    current = []
    prev_t = None
    for d in data:
        t = d["timestep"]
        if prev_t is not None and t > prev_t + 50:
            if current:
                trajectories.append(current)
            current = []
        current.append(d)
        prev_t = t
    if current:
        trajectories.append(current)

    # Group by lambda → one line per lambda value
    lambda_trajs = {}
    for traj in trajectories:
        if not traj:
            continue
        lam_key = f"\u03bb={traj[0]['lambda']:.2f}"
        lambda_trajs[lam_key] = traj  # last trajectory per lambda wins

    if lambda_trajs:
        keys = sorted(lambda_trajs.keys())
        xs = [[d["timestep"] for d in lambda_trajs[k]] for k in keys]
        ys = [[d["guidance_scale"] for d in lambda_trajs[k]] for k in keys]
        _run.log({
            "charts/guidance_scale_trajectories": wandb.plot.line_series(
                xs=xs, ys=ys, keys=keys,
                title="Guidance Scale vs Timestep (Sampling)",
                xname="Timestep",
            ),
        })

    # Also log raw table for custom analysis
    rows = []
    for gen_idx, traj in enumerate(trajectories):
        for step_idx, d in enumerate(traj):
            rows.append([gen_idx, step_idx, d["timestep"], d["guidance_scale"], d["lambda"]])
    if rows:
        table = wandb.Table(data=rows, columns=["generation", "step", "timestep", "guidance_scale", "lambda"])
        _run.log({"sampling/guidance_data": table})
