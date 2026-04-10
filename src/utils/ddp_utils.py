"""DDP utilities — call setup() early, wrap(model) after model creation.

Usage in train.py (2 lines only):
  Line 1 (after imports):  import src.utils.ddp_utils as ddp_utils; ddp_utils.setup()
  Line 2 (after model load): guidance_scale_network = ddp_utils.wrap(guidance_scale_network)

Launch with: torchrun --nproc_per_node=2 scripts/train.py
When LOCAL_RANK is not set (normal python), everything is a no-op.
"""
import datetime
import os
import torch
import torch.distributed as dist

_rank = 0
_local_rank = 0
_world_size = 1
_skip_remaining = 0
_dummy_cache = None


def is_main():
    return _rank == 0


def setup():
    """Initialize DDP if LOCAL_RANK is set (by torchrun). No-op otherwise."""
    global _rank, _local_rank, _world_size
    lr = os.environ.get("LOCAL_RANK")
    if lr is None:
        return

    _local_rank = int(lr)
    _rank = int(os.environ.get("RANK", _local_rank))
    _world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Pin each rank to its own GPU. set_device makes torch.device("cuda") use this GPU.
    torch.cuda.set_device(_local_rank)

    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))

    _patch_dataloader()
    _patch_wandb()
    _patch_checkpoint()
    _patch_auto_sample()
    if not is_main():
        _suppress_prints()


def wrap(model):
    """Wrap model in DDP if distributed is initialized. No-op otherwise."""
    if not dist.is_initialized():
        return model
    return torch.nn.parallel.DistributedDataParallel(model, device_ids=[_local_rank])


# ---------------------------------------------------------------------------
# Monkey-patches (applied once in setup)
# ---------------------------------------------------------------------------

def _patch_dataloader():
    """Replace get_data_loader to use DistributedSampler."""
    import src.utils.train_utils as train_utils
    _orig = train_utils.get_data_loader

    def _ddp_get_data_loader(config):
        from torch.utils.data import DataLoader, DistributedSampler
        from src.data.dataset import LaionDataset

        batch_size = config["training"]["batch_size"]
        image_root = config["training"]["image_root"]
        dataset = LaionDataset(image_root)
        sampler = DistributedSampler(
            dataset, num_replicas=_world_size, rank=_rank, shuffle=True,
        )
        dl = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, pin_memory=True,
        )
        dl.distributed_sampler = sampler
        return dl

    train_utils.get_data_loader = _ddp_get_data_loader


def _patch_wandb():
    """Make W&B a no-op on non-main ranks."""
    if is_main():
        return
    import src.utils.wandb_utils as wb
    wb.init_training = lambda *a, **kw: None
    wb.log_train = lambda *a, **kw: None
    wb.finish = lambda *a, **kw: None


def _patch_checkpoint():
    """Only save checkpoints on rank 0; unwrap DDP module for clean state_dict.
    Also patch maybe_resume to load into the inner module."""
    import src.utils.resume_utils as resume_utils
    _orig_save = resume_utils.save_checkpoint
    _orig_resume = resume_utils.maybe_resume

    def _ddp_save(config, model, optimizer, step, timestamp, *args, **kwargs):
        if not is_main():
            return
        raw = model.module if hasattr(model, "module") else model
        _orig_save(config, raw, optimizer, step, timestamp, *args, **kwargs)

    def _ddp_resume(config, model, optimizer=None):
        global _skip_remaining
        raw = model.module if hasattr(model, "module") else model
        step = _orig_resume(config, raw, optimizer)
        if step > 0:
            _skip_remaining = step
            _patch_dataset_skip()
        return step

    resume_utils.save_checkpoint = _ddp_save
    resume_utils.maybe_resume = _ddp_resume


def _patch_auto_sample():
    """Only run auto-sampling on rank 0."""
    try:
        import src.utils.train_utils_sd3 as tu3
    except ImportError:
        return
    if not hasattr(tu3, "run_auto_sample"):
        return
    _orig = tu3.run_auto_sample

    def _guarded(*a, **kw):
        if is_main():
            return _orig(*a, **kw)

    tu3.run_auto_sample = _guarded


def _patch_dataset_skip():
    """Monkey-patch LaionDataset.__getitem__ to return cached dummy data during
    the resume skip phase, avoiding expensive disk I/O for images that will be
    discarded by the `global_step < resume_step` check in train()."""
    from src.data.dataset import LaionDataset
    _orig_getitem = LaionDataset.__getitem__

    def _fast_getitem(self, idx):
        global _skip_remaining, _dummy_cache
        if _skip_remaining > 0:
            _skip_remaining -= 1
            if _dummy_cache is not None:
                return _dummy_cache
            # First call: load one real sample and cache it
            result = _orig_getitem(self, idx)
            _dummy_cache = result
            return result
        return _orig_getitem(self, idx)

    LaionDataset.__getitem__ = _fast_getitem


def _suppress_prints():
    """Silence stdout on non-main ranks."""
    import builtins
    builtins.print = lambda *a, **kw: None
