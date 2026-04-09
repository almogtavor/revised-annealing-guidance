import torch
import torch.nn as nn
from typing import Union


class ScalarMLP(nn.Module):
    """
    MLP that outputs per-example guidance scales given (timestep, lambda, delta),
    where delta = (noise_pred_uncond - noise_pred_text) and delta_norm = ||delta||_2 per example.

    forward(...) returns guidance scales.
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        n_layers: int,
        # Embedding sizes
        t_embed_dim: int = 4,
        delta_embed_dim: int = 4,
        lambda_embed_dim: int = 4,
        interval_embed_dim: int = 0,  # 0 = disabled (backward compatible)
        c_embed_dim: int = 0,         # 0 = disabled; projected from pooled prompt embeds
        c_input_dim: int = 2048,      # SD3 pooled prompt embed dim (CLIP-L 768 + CLIP-G 1280)
        # Normalizations applied before embedding
        t_embed_normalization: float = 1e3,  # SD3 timesteps are [0, 1000], so t/1000 → [0, 1]
        delta_embed_normalization: float = 5.0,
        # Number of denoising steps (T); timestep input is normalized as t/T
        num_timesteps: int = None,
        # Final affine on head output
        w_bias: float = 1.0,
        w_scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()

        input_size = t_embed_dim + delta_embed_dim + lambda_embed_dim + interval_embed_dim + c_embed_dim

        # Head
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(n_layers):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers += [nn.Linear(hidden_size, output_size)]
        self.combined_head = nn.Sequential(*layers)

        # Prompt projection (only when c_embed_dim > 0)
        self.c_proj = nn.Linear(c_input_dim, c_embed_dim) if c_embed_dim > 0 else None

        # Config
        self.t_embed_dim = t_embed_dim
        self.delta_embed_dim = delta_embed_dim
        self.lambda_embed_dim = lambda_embed_dim
        self.interval_embed_dim = interval_embed_dim
        self.c_embed_dim = c_embed_dim

        # Always normalize timesteps to [0, 1] via t/1000 (SD3 timesteps are in [0, 1000])
        self.t_embed_normalization = t_embed_normalization
        self.delta_embed_normalization = delta_embed_normalization

        self.w_bias = w_bias
        self.w_scale = w_scale

    # ---------- helpers ----------

    @staticmethod
    def _ensure_batched(x: Union[float, int, torch.Tensor], B: int, device, dtype) -> torch.Tensor:
        """Make x a length-B tensor on (device, dtype)."""
        x = torch.as_tensor(x, device=device, dtype=dtype)
        return x.expand(B) if x.dim() == 0 else x

    @staticmethod
    def _embed_value(value: torch.Tensor, n_embeds: int) -> torch.Tensor:
        """
        Positional-like embedding: [value, cos(value*1), ..., cos(value*(n_embeds-1))].
        Expects value shape: (B,)
        Returns: (B, n_embeds)
        """
        i = torch.arange(1, n_embeds, device=value.device, dtype=value.dtype)
        cosines = torch.cos(value.unsqueeze(-1) * i)
        return torch.cat([value.unsqueeze(-1), cosines], dim=-1)

    # ---------- forward ----------

    def forward(
        self,
        timestep: Union[float, int, torch.Tensor],
        l: Union[float, int, torch.Tensor],
        noise_pred_uncond: torch.Tensor,
        noise_pred_text: torch.Tensor,
        interval: Union[float, int, torch.Tensor] = 0.0,
        c_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            timestep: scalar or (B,)
            l:        scalar or (B,)
            noise_pred_uncond: (B, C, H, W)
            noise_pred_text:   (B, C, H, W)
            interval: (t-s)/T normalized interval length, scalar or (B,).
                      Only used when interval_embed_dim > 0; ignored otherwise.
            c_emb:    (B, c_input_dim) pooled prompt embeddings.
                      Only used when c_embed_dim > 0; ignored otherwise.

        Returns:
            guidance_scales: (B,) if output_size==1 else (B, output_size)
        """
        # 1) Compute delta and its norm per example
        delta = noise_pred_uncond - noise_pred_text            # (B, C, H, W)
        B = delta.shape[0]
        delta_norm = delta.view(B, -1).norm(dim=1)             # (B,)

        # 2) Unify device/dtype & batch shapes for scalar inputs
        device, dtype = delta_norm.device, delta_norm.dtype
        timestep = self._ensure_batched(timestep, B, device, dtype)  # (B,)
        l        = self._ensure_batched(l,        B, device, dtype)  # (B,)

        # 3) Build features with embeddings
        t_feat = self._embed_value(timestep / self.t_embed_normalization, self.t_embed_dim)           # (B, t_embed_dim)
        d_feat = self._embed_value(delta_norm / self.delta_embed_normalization, self.delta_embed_dim) # (B, delta_embed_dim)
        l_feat = self._embed_value(l, self.lambda_embed_dim)        # (B, lambda_embed_dim)

        parts = [t_feat, d_feat, l_feat]

        if self.interval_embed_dim > 0:
            interval = self._ensure_batched(interval, B, device, dtype)
            iv_feat = self._embed_value(interval, self.interval_embed_dim)  # (B, interval_embed_dim)
            parts.append(iv_feat)

        if self.c_proj is not None and c_emb is not None:
            c_feat = self.c_proj(c_emb.to(dtype=dtype, device=device))  # (B, c_embed_dim)
            parts.append(c_feat)

        features = torch.cat(parts, dim=-1)  # (B, input_size)

        # 4) Head → scales
        guidance_scale = self.combined_head(features)                   # (B, output_size)
        guidance_scale = self.w_scale * guidance_scale + self.w_bias

        # Squeeze to (B,) for single-output models
        if guidance_scale.shape[1] == 1:
            guidance_scale = guidance_scale.squeeze(-1)
        return guidance_scale


def mlp_extras(model, t, next_t, pooled_prompt_embeds):
    """Build extra kwargs for extended-MLP guidance models trained with interval & c_emb conditioning.

    Args:
        model: ScalarMLP (possibly wrapped in AutoLambdaWrapper).
        t: current timestep tensor.
        next_t: next timestep tensor (or 0-tensor at the final step).
        pooled_prompt_embeds: doubled [uncond, cond] pooled prompt embeds; the cond half is used.
    """
    kw = {}
    inner = getattr(model, 'mlp', model)  # unwrap AutoLambdaWrapper if any
    if getattr(inner, 'interval_embed_dim', 0) > 0:
        kw['interval'] = (t.float() - next_t.float()) / 1000.0
    if getattr(inner, 'c_embed_dim', 0) > 0:
        # pooled_prompt_embeds is doubled [uncond, cond]; take cond half
        kw['c_emb'] = pooled_prompt_embeds[pooled_prompt_embeds.shape[0] // 2:].float()
    return kw
