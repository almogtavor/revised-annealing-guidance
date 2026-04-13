"""FSG (Fixed-point Stochastic Guidance) training utilities.

Implements the FSG-aligned forward pass for SD3 flow-matching training.
The FSG block runs K inner iterations of (guided forward → unconditional inverse)
at the first 3 early denoising sites, then does a final calibration step.

Training is split 50/50 between FSG-aligned and regular one-step modes.
"""

import torch
import torch.distributed as dist
import src.utils.train_utils as train_utils
import src.utils.train_utils_sd3 as train_utils_sd3

# FSG schedule: 3 high-noise sites with decreasing iterations
# Interval Δ = 0.125 * 1000 (covers the first 0.375 of the denoising trajectory)
FSG_DELTA_FRAC = 0.125     # Δ/T
FSG_NUM_SITES = 3          # first 3 sites
FSG_ITERATIONS = [3, 2, 2] # K per site
FSG_SC_WEIGHT = 0.1        # self-consistency MMD weight


def _gather_particles(x, keep_grad=False):
    if not dist.is_available() or not dist.is_initialized():
        return x
    gathered = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, x.detach())
    if keep_grad:
        gathered[dist.get_rank()] = x
    return torch.cat(gathered, dim=0)


def _energy_mmd(x, y):
    """Mini-batch MMD with the paper's energy kernel k(x, y) = -||x-y||_2."""
    x = _gather_particles(x.float(), keep_grad=True).flatten(1)
    y = _gather_particles(y.float()).flatten(1)

    cross = torch.cdist(x, y).mean()
    same_x = torch.cdist(x, x).mean()
    same_y = torch.cdist(y, y).mean()
    return 2.0 * cross - same_x - same_y


def get_fsg_sites(T):
    """Return the 3 earliest high-noise FSG sites in [0, 1000] time.

    The sites are anchored at t = 1000, 875, 750 with interval Δt = 125.
    Site 0 is the noisiest and fires first during denoising.
    """
    _ = T  # site placement is defined in normalized diffusion time, not by NFE
    delta_t = FSG_DELTA_FRAC * 1000.0
    sites = []
    for i in range(FSG_NUM_SITES):
        t_site = 1000.0 - i * delta_t
        t_target = max(t_site - delta_t, 0.0)
        sites.append({
            'index': i,
            't': t_site,
            's': t_target,
            'sigma': t_site / 1000.0,
            'sigma_s': t_target / 1000.0,
            'K': FSG_ITERATIONS[i],
        })
    return sites


def snap_to_fsg_site(timestep_sigma, T):
    """Snap a sigma value to the nearest FSG site. Returns (site_dict, distance) or (None, inf)."""
    sites = get_fsg_sites(T)
    best_site, best_dist = None, float('inf')
    fsg_boundary = 1.0 - FSG_NUM_SITES * FSG_DELTA_FRAC  # 0.625
    for site in sites:
        dist = abs(timestep_sigma - site['sigma'])
        if dist < best_dist:
            best_site, best_dist = site, dist
    # Only snap if within the early high-noise FSG region.
    if timestep_sigma >= fsg_boundary:
        return best_site, best_dist
    return None, float('inf')


def fsg_inner_loop(pipeline, model, z_t, t_val, s_val, sigma_t, sigma_s,
                   pe, ppe, lam, K, interval_norm, dtype, c_emb=None):
    """Run K FSG inner iterations: guided forward → unconditional inverse.

    Args:
        pipeline: SD3 pipeline (frozen transformer)
        model: ScalarMLP (trainable)
        z_t: current latent at t (B, C, H, W)
        t_val, s_val: timestep values (float, in [0, 1000])
        sigma_t, sigma_s: sigma values (float, in [0, 1])
        pe, ppe: prompt embeddings (already doubled for CFG)
        lam: lambda values (B,)
        K: number of inner iterations
        interval_norm: (t-s)/T normalized interval for MLP
        dtype: pipeline dtype
        c_emb: (B, c_input_dim) pooled prompt embeddings for MLP conditioning
    Returns:
        z_t_refined: refined latent at t after K iterations
    """
    device = z_t.device
    B = z_t.shape[0]

    # Half embeddings for unconditional-only forward pass
    pe_uncond = pe[:pe.shape[0] // 2]
    ppe_uncond = ppe[:ppe.shape[0] // 2]

    t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)
    s_tensor = torch.full((B,), s_val, device=device, dtype=torch.long)

    for _k in range(K):
        # (1) Compute v_u, v_c at z_t
        with torch.no_grad():
            pred = train_utils_sd3.denoise_single_step_sd3(
                pipeline, z_t.to(dtype), pe, ppe, t_tensor)
        vu, vt = pred.float().chunk(2)
        del pred

        # (2) Predict w with interval and prompt awareness
        w = model(t_tensor.float(), lam, vu, vt, interval=interval_norm, c_emb=c_emb)
        v_guided = vu + w.view(-1, 1, 1, 1) * (vt - vu)

        # (3) Guided forward step: z_t → z_s
        z_t_f32 = z_t.float()
        x0 = z_t_f32 - sigma_t * v_guided
        eps_u = z_t_f32 + (1.0 - sigma_t) * vu
        z_s = (1.0 - sigma_s) * x0 + sigma_s * eps_u

        # (4) Unconditional inverse step: z_s → z_t
        with torch.no_grad():
            # Single forward (uncond only, no CFG doubling)
            vu_s = pipeline.transformer(
                hidden_states=z_s.to(dtype),
                timestep=s_tensor.expand(B),
                encoder_hidden_states=pe_uncond,
                pooled_projections=ppe_uncond,
                return_dict=False,
            )[0].float()

        # Inverse flow: z_s at sigma_s → z_t at sigma_t
        x0_inv = z_s.float() - sigma_s * vu_s
        eps_inv = z_s.float() + (1.0 - sigma_s) * vu_s
        z_t = ((1.0 - sigma_t) * x0_inv + sigma_t * eps_inv).detach()
        # Detach intermediate z_t to avoid backprop through all K iterations
        # (self-consistency loss will be on the final state only)

    return z_t


_fsg_global_images = 0  # updated by train loop


def forward_pass_fsg(config, pipeline, model, images, prompts, image_paths=None):
    """FSG-aligned SD3 forward pass.

    50% chance: regular one-step training (identical to forward_pass_sd3)
    50% chance: FSG-aligned training
      - if t is in FSG region [0, 0.375T]: snap to nearest site, run FSG block
      - otherwise: regular one-step training
    Supports fsg.start_after_images: skip FSG branch until enough images seen.
    """
    B = images.size(0)
    dtype = pipeline.transformer.dtype
    T = train_utils_sd3.get_num_sampling_steps(config, default=20)
    fsg_cfg = config.get('fsg', {})
    sc_weight = fsg_cfg.get('sc_weight', FSG_SC_WEIGHT)
    fsg_active = _fsg_global_images >= fsg_cfg.get('start_after_images', 0)

    lam = torch.rand(B).to(pipeline.device)
    fixed_lam = config['training'].get('fixed_lambda')
    if fixed_lam is not None:
        lam = torch.full_like(lam, fixed_lam)

    timestep = train_utils.get_timestep(pipeline, batch_size=B)
    noisy_latents, velocity_gt = train_utils_sd3.to_noisy_latents_sd3(pipeline, images, timestep)

    cache_dir = config['training'].get('prompt_cache_dir')
    if cache_dir and image_paths:
        pe, ppe = train_utils_sd3.load_cached_prompt_sd3(
            cache_dir, config['training']['image_root'], image_paths, pipeline.device)
    else:
        with torch.no_grad():
            pe, ppe = train_utils_sd3.encode_prompt_sd3(pipeline, prompts)
    pe, ppe = train_utils_sd3.prompt_add_noise_sd3(
        pe, ppe, timestep, pipeline.scheduler.config.get('num_train_timesteps', 1000),
        **config['training']['prompt_noise'])

    # Extract conditional pooled prompt embeddings for MLP (second half of doubled ppe)
    c_emb = ppe[ppe.shape[0] // 2:].float()  # (B, 2048)

    # Decide mode: 50% FSG-aligned, 50% regular (only if FSG is active)
    use_fsg_mode = fsg_active and torch.rand(1).item() < 0.5

    # Check if timestep falls in the early high-noise FSG region [0.625 * 1000, 1000]
    t_val = timestep.float().mean().item()
    fsg_boundary = (1.0 - FSG_NUM_SITES * FSG_DELTA_FRAC) * 1000.0  # 625.0

    # If coin says FSG and t is in the early region, snap to nearest FSG site
    run_fsg_block = use_fsg_mode and t_val >= fsg_boundary
    if run_fsg_block:
        sigma_val = t_val / 1000.0
        fsg_site, _ = snap_to_fsg_site(sigma_val, T)

    if run_fsg_block:
        # --- FSG-aligned training ---
        site = fsg_site
        interval_norm = (site['t'] - site['s']) / 1000.0  # (t-s)/t_embed_normalization

        # Snap timestep to site
        t_snapped = torch.full_like(timestep, int(site['t']))
        sigma_t = site['sigma']
        sigma_s = site['sigma_s']

        # Re-noise latents at the snapped timestep
        noisy_latents_snapped, velocity_gt_snapped = train_utils_sd3.to_noisy_latents_sd3(
            pipeline, images, t_snapped)

        # Re-encode prompts at snapped timestep
        pe_snap, ppe_snap = pe, ppe  # reuse (prompt noise was already applied)

        # Run FSG inner loop
        z_t_refined = fsg_inner_loop(
            pipeline, model, noisy_latents_snapped, site['t'], site['s'],
            sigma_t, sigma_s, pe_snap, ppe_snap, lam, site['K'],
            interval_norm, dtype, c_emb=c_emb)

        # Final: recompute guidance at refined z_t
        with torch.no_grad():
            pred_final = train_utils_sd3.denoise_single_step_sd3(
                pipeline, z_t_refined.to(dtype), pe_snap, ppe_snap, t_snapped)
        vu_final, vt_final = pred_final.float().chunk(2)

        # Match training/inference to the scheduler's discrete timestep grid.
        t_next_val = train_utils_sd3.get_prev_timestep(pipeline.scheduler, t_snapped)
        st = (t_snapped.float() / 1000.0).to(device=z_t_refined.device)
        st1 = (t_next_val.float() / 1000.0).to(device=z_t_refined.device)
        interval = (t_snapped.float() - t_next_val.float()) / 1000.0  # (t - s) / t_embed_normalization

        w_final = model(t_snapped.float(), lam, vu_final, vt_final, interval=interval, c_emb=c_emb)
        v_guided = vu_final + w_final.view(-1, 1, 1, 1) * (vt_final - vu_final)

        # CFG++ step from snapped site
        st_, st1_ = st.view(-1, 1, 1, 1), st1.view(-1, 1, 1, 1)
        zf = z_t_refined.float()
        x0 = zf - st_ * v_guided
        eps_u = zf + (1.0 - st_) * vu_final
        z_next = (1.0 - st1_) * x0 + st1_ * eps_u
        del x0, eps_u, zf
        torch.cuda.empty_cache()

        pred2 = train_utils_sd3.denoise_single_step_sd3(
            pipeline, z_next.to(dtype), pe_snap, ppe_snap, t_next_val)
        vu2, vt2 = pred2.float().chunk(2)
        delta = vt2 - vu2

        # Losses
        ema_normalize = config['training'].get('ema_loss_normalization', True)
        loss = train_utils.calc_loss(v_guided, velocity_gt_snapped.float(), delta, lam,
                                     ema_normalize=ema_normalize)

        # Self-consistency on the final FSG iterate: match guided and noised site marginals.
        sc_loss = _energy_mmd(z_t_refined, noisy_latents_snapped)
        # loss = loss + sc_weight * sc_loss  # disabled: SC not included in loss

        eps_val = ((1 - lam) * ((v_guided - velocity_gt_snapped.float()) ** 2).mean(dim=[1, 2, 3])).mean()
        diff_val = (lam * (delta ** 2).mean(dim=[1, 2, 3])).mean()
        delta_t = (vt_final - vu_final)
        delta_norm = delta_t.view(B, -1).norm(dim=1).mean().item()
        delta_next_norm = delta.view(B, -1).norm(dim=1).mean().item()

        return {
            'loss': loss,
            'train/eps_loss': eps_val.item(),
            'train/diff_loss': diff_val.item(),
            'train/delta_norm': delta_norm,
            'train/delta_next_norm': delta_next_norm,
            'train/fsg_mode': 1.0,
            'train/fsg_site': float(site['index']),
            'train/fsg_K': float(site['K']),
            'train/sc_loss': sc_loss.item(),
            'train/w_mean': w_final.detach().mean().item(),
        }

    else:
        # --- Regular one-step training ---
        with torch.no_grad():
            pred = train_utils_sd3.denoise_single_step_sd3(
                pipeline, noisy_latents, pe, ppe, timestep)
        vu, vt = pred.float().chunk(2)
        del pred

        # Use the scheduler's actual previous timestep, not a uniform 1/T decrement.
        t_next = train_utils_sd3.get_prev_timestep(pipeline.scheduler, timestep)
        st = (timestep.float() / 1000.0).to(device=noisy_latents.device)
        st1 = (t_next.float() / 1000.0).to(device=noisy_latents.device)
        interval = (timestep.float() - t_next.float()) / 1000.0  # (t - s) / t_embed_normalization

        w = model(timestep.float(), lam, vu, vt, interval=interval, c_emb=c_emb)
        v_guided = vu + w.view(-1, 1, 1, 1) * (vt - vu)

        st_, st1_ = st.view(-1, 1, 1, 1), st1.view(-1, 1, 1, 1)
        zf = noisy_latents.float()
        x0 = zf - st_ * v_guided
        eps_u = zf + (1.0 - st_) * vu
        z_next = (1.0 - st1_) * x0 + st1_ * eps_u
        del x0, eps_u, zf
        torch.cuda.empty_cache()

        pred2 = train_utils_sd3.denoise_single_step_sd3(
            pipeline, z_next.to(dtype), pe, ppe, t_next)
        vu2, vt2 = pred2.float().chunk(2)
        delta = vt2 - vu2

        ema_normalize = config['training'].get('ema_loss_normalization', True)
        loss = train_utils.calc_loss(v_guided, velocity_gt.float(), delta, lam,
                                     ema_normalize=ema_normalize)

        eps_val = ((1 - lam) * ((v_guided - velocity_gt.float()) ** 2).mean(dim=[1, 2, 3])).mean()
        diff_val = (lam * (delta ** 2).mean(dim=[1, 2, 3])).mean()
        delta_t = (vt - vu)
        delta_norm = delta_t.view(B, -1).norm(dim=1).mean().item()
        delta_next_norm = delta.view(B, -1).norm(dim=1).mean().item()

        return {
            'loss': loss,
            'train/eps_loss': eps_val.item(),
            'train/diff_loss': diff_val.item(),
            'train/delta_norm': delta_norm,
            'train/delta_next_norm': delta_next_norm,
            'train/fsg_mode': 0.0,
            'train/fsg_site': -1.0,
            'train/fsg_K': 0.0,
            'train/sc_loss': 0.0,
            'train/w_mean': w.detach().mean().item(),
        }
