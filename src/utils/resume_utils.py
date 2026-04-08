"""Checkpoint resumption utilities for preemptible SLURM training."""
import os
import re
import glob
import torch


def maybe_resume(config, model, optimizer=None):
    """Load checkpoint only if 'resume_from' is set in config.training.

    Set  training.resume_from: path/to/checkpoint_step_XXXX.pt  in the YAML
    to resume. If the key is absent or null, training starts from scratch.

    Returns the step to resume from (0 if no resume).
    """
    ckpt_path = config.get('training', {}).get('resume_from')
    if not ckpt_path:
        return 0

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"resume_from checkpoint not found: {ckpt_path}")

    print(f"Resuming from checkpoint: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # Support both old ('model_state_dict') and new ('guidance_scale_model') key names
    state_key = 'guidance_scale_model' if 'guidance_scale_model' in ckpt else 'model_state_dict'
    model.load_state_dict(ckpt[state_key])

    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print("  Restored optimizer state.", flush=True)

    step = ckpt.get('step', 0)
    print(f"  Resuming from step {step}.", flush=True)
    return step


def save_checkpoint(config, model, optimizer, step, timestamp):
    """Save checkpoint with optimizer state for resumption."""
    out_dir = config['training']['out_dir']
    checkpoint_dir = f'{out_dir}/checkpoints_{timestamp}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({
        'config': config,
        'guidance_scale_model': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
    }, f'{checkpoint_dir}/checkpoint_step_{step}.pt')
