# This file includes and modifies code derived from:
#   - "pipeline_stable_diffusion3" from Hugging Face Diffusers
#     Copyright 2024 The HuggingFace Team
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#         http://www.apache.org/licenses/LICENSE-2.0
#

import torch
from typing import Any, Callable, Dict, List, Optional, Union

from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
    calculate_shift,
)
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.utils import is_torch_xla_available

from src.model.guidance_scale_model import ScalarMLP

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class MyStableDiffusion3Pipeline(StableDiffusion3Pipeline):
    """
    Custom SD3 pipeline that adds optional annealing guidance scale support.

    Only the CFG computation in __call__ is modified; everything else is
    inherited from StableDiffusion3Pipeline.
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        skip_guidance_layers: List[int] = None,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
        mu: Optional[float] = None,
        # --- NEW: Annealing guidance parameters ---
        use_annealing_guidance: bool = False,
        guidance_scale_model: Optional[ScalarMLP] = None,
        guidance_lambda: Optional[float] = None,
        use_cfgpp: bool = False,
        # --- FSG: Fixed-point Stochastic Guidance ---
        use_fsg: bool = False,
        fsg_iterations: int = 3,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._cfgpp_w = guidance_scale if use_cfgpp else None
        # CFG++ with w<1 still needs both cond/uncond predictions
        if use_cfgpp and guidance_scale <= 1.0:
            self._guidance_scale = 1.1  # force do_classifier_free_guidance=True
        self._skip_layer_guidance_scale = skip_layer_guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            if skip_guidance_layers is not None:
                original_prompt_embeds = prompt_embeds
                original_pooled_prompt_embeds = pooled_prompt_embeds
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, height, width = latents.shape
            image_seq_len = (height // self.transformer.config.patch_size) * (
                width // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Prepare image embeddings
        if (ip_adapter_image is not None and self.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
            ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

            if self.joint_attention_kwargs is None:
                self._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
            else:
                self._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                _use_cfgpp_step = False
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                    # --- BEGIN MODIFIED: Annealing guidance / CFG++ / CFG ---
                    if use_annealing_guidance and guidance_scale_model is not None:
                        # Learned annealing guidance (CFG++ sampling)
                        orig_dtype = noise_pred_uncond.dtype
                        guidance_scale_pred = guidance_scale_model(
                            t, guidance_lambda,
                            noise_pred_uncond.float(), noise_pred_text.float()
                        )
                        v_guided = noise_pred_uncond.float() + guidance_scale_pred * (noise_pred_text.float() - noise_pred_uncond.float())
                        _use_cfgpp_step = True
                    elif use_cfgpp:
                        # Fixed-w CFG++ sampling
                        orig_dtype = noise_pred_uncond.dtype
                        v_guided = noise_pred_uncond.float() + self._cfgpp_w * (noise_pred_text.float() - noise_pred_uncond.float())
                        _use_cfgpp_step = True
                    else:
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if _use_cfgpp_step:
                        sigma_t = t.float() / 1000.0
                        sigma_t1 = timesteps[i + 1].float() / 1000.0 if (i + 1 < len(timesteps)) else 0.0
                        orig_dtype = noise_pred_uncond.dtype

                        if use_fsg and fsg_iterations > 0:
                            # --- FSG: Fixed-point Stochastic Guidance ---
                            # K inner iterations: guided forward → unconditional inverse
                            # to refine z_t before the final step.
                            z_t = latents.float()
                            for _k in range(fsg_iterations):
                                # (1) Compute v_u, v_c, delta at current z_t
                                _fsg_input = torch.cat([z_t.to(orig_dtype)] * 2)
                                _fsg_ts = t.expand(_fsg_input.shape[0])
                                _fsg_pred = self.transformer(
                                    hidden_states=_fsg_input,
                                    timestep=_fsg_ts,
                                    encoder_hidden_states=prompt_embeds,
                                    pooled_projections=pooled_prompt_embeds,
                                    joint_attention_kwargs=self.joint_attention_kwargs,
                                    return_dict=False,
                                )[0]
                                _vu, _vt = _fsg_pred.chunk(2)
                                del _fsg_pred, _fsg_input

                                # (2) Predict w and form guided velocity
                                if use_annealing_guidance and guidance_scale_model is not None:
                                    _w = guidance_scale_model(t, guidance_lambda, _vu.float(), _vt.float())
                                    _v_guided = _vu.float() + _w * (_vt.float() - _vu.float())
                                else:
                                    _v_guided = _vu.float() + self._cfgpp_w * (_vt.float() - _vu.float())

                                # (3) Guided forward step: z_t → z_s (using sigma_t1 as s)
                                _x0 = z_t - sigma_t * _v_guided
                                _eps_u = z_t + (1.0 - sigma_t) * _vu.float()
                                z_s = (1.0 - sigma_t1) * _x0 + sigma_t1 * _eps_u
                                del _x0, _eps_u, _v_guided, _vu, _vt

                                # (4) Unconditional inverse step: z_s → z_t
                                _fsg_inv_ts = timesteps[i + 1] if (i + 1 < len(timesteps)) else t
                                _fsg_inv_ts = _fsg_inv_ts.expand(z_s.shape[0])
                                _vu_s = self.transformer(
                                    hidden_states=z_s.to(orig_dtype),
                                    timestep=_fsg_inv_ts,
                                    encoder_hidden_states=prompt_embeds[:prompt_embeds.shape[0]//2],
                                    pooled_projections=pooled_prompt_embeds[:pooled_prompt_embeds.shape[0]//2],
                                    joint_attention_kwargs=self.joint_attention_kwargs,
                                    return_dict=False,
                                )[0].float()

                                # Inverse flow: z_s at sigma_s → z_t at sigma_t
                                _x0_inv = z_s - sigma_t1 * _vu_s
                                _eps_inv = z_s + (1.0 - sigma_t1) * _vu_s
                                z_t = (1.0 - sigma_t) * _x0_inv + sigma_t * _eps_inv
                                del z_s, _vu_s, _x0_inv, _eps_inv
                                torch.cuda.empty_cache()

                            # Final: recompute guidance at refined z_t and do actual step
                            _fsg_input = torch.cat([z_t.to(orig_dtype)] * 2)
                            _fsg_ts = t.expand(_fsg_input.shape[0])
                            _fsg_pred = self.transformer(
                                hidden_states=_fsg_input,
                                timestep=_fsg_ts,
                                encoder_hidden_states=prompt_embeds,
                                pooled_projections=pooled_prompt_embeds,
                                joint_attention_kwargs=self.joint_attention_kwargs,
                                return_dict=False,
                            )[0]
                            noise_pred_uncond, noise_pred_text = _fsg_pred.chunk(2)
                            if use_annealing_guidance and guidance_scale_model is not None:
                                _w = guidance_scale_model(t, guidance_lambda, noise_pred_uncond.float(), noise_pred_text.float())
                                v_guided = noise_pred_uncond.float() + _w * (noise_pred_text.float() - noise_pred_uncond.float())
                            else:
                                v_guided = noise_pred_uncond.float() + self._cfgpp_w * (noise_pred_text.float() - noise_pred_uncond.float())
                            latents_f32 = z_t
                        else:
                            latents_f32 = latents.float()

                        # CFG++ denoising step (flow-matching equivalent)
                        x0_pred = latents_f32 - sigma_t * v_guided
                        eps_uncond = latents_f32 + (1.0 - sigma_t) * noise_pred_uncond.float()
                        latents = ((1.0 - sigma_t1) * x0_pred + sigma_t1 * eps_uncond).to(orig_dtype)

                        # Advance scheduler step index (we bypassed scheduler.step)
                        if self.scheduler.step_index is None:
                            self.scheduler._init_step_index(t)
                        self.scheduler._step_index += 1
                    # --- END MODIFIED ---

                    should_skip_layers = (
                        True
                        if i > num_inference_steps * skip_layer_guidance_start
                        and i < num_inference_steps * skip_layer_guidance_stop
                        else False
                    )
                    if skip_guidance_layers is not None and should_skip_layers:
                        timestep = t.expand(latents.shape[0])
                        latent_model_input = latents
                        noise_pred_skip_layers = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=original_prompt_embeds,
                            pooled_projections=original_pooled_prompt_embeds,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                            skip_layers=skip_guidance_layers,
                        )[0]
                        noise_pred = (
                            noise_pred + (noise_pred_text - noise_pred_skip_layers) * self._skip_layer_guidance_scale
                        )

                # compute the previous noisy sample x_t -> x_t-1
                if not _use_cfgpp_step:
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                latents_dtype = latents.dtype

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    pooled_prompt_embeds = callback_outputs.pop("pooled_prompt_embeds", pooled_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)
