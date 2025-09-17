"""
model evaluation/sampling
"""
import math
import os
import pathlib

import torch
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from dataclasses import asdict

import diffusers
from diffusers import DiffusionPipeline, ImagePipelineOutput, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
import shutil

from utils_diff import make_grid
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


####################
# modified from segmentation-guided DDPM
####################


def evaluate_sample_many(sample_size,
                         config,
                         model,
                         noise_scheduler,
                         eval_dataloader,
                         device='cuda'):
    # for loading segs to condition on:
    # setup for sampling
    if config.model_type == "DDPM":
        if config.segmentation_guided:
            pipeline = SegGuidedDDPMPipeline(unet=model.module,
                                             scheduler=noise_scheduler,
                                             eval_dataloader=eval_dataloader,
                                             external_config=config)
        else:
            pipeline = diffusers.DDPMPipeline(unet=model.module,
                                              scheduler=noise_scheduler)
    elif config.model_type == "DDIM":
        if config.segmentation_guided:
            pipeline = SegGuidedDDIMPipeline(unet=model.module,
                                             scheduler=noise_scheduler,
                                             eval_dataloader=eval_dataloader,
                                             external_config=config)
        else:
            pipeline = diffusers.DDIMPipeline(unet=model.module,
                                              scheduler=noise_scheduler)

    sample_dir = test_dir = os.path.join(config.output_dir,
                                         "samples_many_{}".format(sample_size))
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    num_sampled = 0
    # keep sampling images until we have enough
    for bidx, seg_batch in tqdm(enumerate(eval_dataloader),
                                total=len(eval_dataloader)):
        if num_sampled < sample_size:
            if config.segmentation_guided:
                current_batch_size = [
                    v for k, v in seg_batch.items() if k.startswith("seg_")
                ][0].shape[0]
            else:
                current_batch_size = config.eval_batch_size

            if config.segmentation_guided:
                images = pipeline(
                    batch_size=current_batch_size,
                    seg_batch=seg_batch,
                ).images
            else:
                images = pipeline(batch_size=current_batch_size, ).images

            # --- 2.5D diff ---
            # for i in range(images.shape[0]):
            #     if config.segmentation_guided:
            #         original = seg_batch["image_filenames"][i]
            #         if not isinstance(original, str):
            #             original = str(original)
            #         stem = os.path.splitext(os.path.basename(original))[0]
            #         out_path = os.path.join(sample_dir, f"condon_{stem}.npy")
            #     else:
            #         out_path = os.path.join(sample_dir, f"{num_sampled + i:04d}.npy")
            #     np.save(out_path, images[i])

            for i in range(images.shape[0]):
                if config.segmentation_guided:
                    original = seg_batch["image_filenames"][i]
                    if not isinstance(original, str):
                        original = str(original)

                    # original looks like "volume123_axial_0050of0103"
                    prefix, axial_part = original.split("_axial_")
                    slice_idx = axial_part.split("of")[0]

                    out_name = f"condon_{prefix}_axial_{slice_idx}.npy"
                    out_path = os.path.join(sample_dir, out_name)
                else:
                    out_path = os.path.join(sample_dir,
                                            f"{num_sampled + i:04d}.npy")

                np.save(out_path, images[i])
            # ---

            num_sampled += len(images)
            print("sampled {}/{}.".format(num_sampled, sample_size))


def convert_segbatch_to_multiclass(shape, segmentations_batch, config, device):
    # NOTE: this generic function assumes that segs don't overlap
    # put all segs on same channel
    segs = torch.zeros(shape).to(device)

    # --- 2.5D diff ---
    for k, seg in segmentations_batch.items():
        if not k.startswith("seg_"):
            continue
        seg = seg.to(device)  # [B, K, H, W] (2.5D) or [B, 1, H, W] (2D)

        if config.use_multislice:
            # expect seg: [B,K,H,W]; broadcast if [B,1,H,W]
            if seg.dim() == 4 and seg.size(1) == 1:
                seg = seg.expand(-1, shape[1], -1, -1)
            elif seg.dim() == 4 and seg.size(1) != shape[1]:
                raise ValueError(
                    f"seg channels ({seg.size(1)}) != K ({shape[1]})")
        else:
            # 2D
            # if seg.dim() == 4 and seg.size(1) != 1:
            #     seg = seg[:, :1, ...]

            segs[segs == 0] = seg[segs == 0]

        # first-nonzero-wins merge
        mask = (segs == 0)
        segs[mask] = seg[mask]
    # ---

    if config.use_ablated_segmentations:
        # randomly remove class labels from segs with some probability
        segs = ablate_masks(segs, config)

    return segs


def add_segmentations_to_noise(noisy_images, segmentations_batch, config,
                               device):
    """
    concat segmentations to noisy image
    """
    # --- 2.5D diff ---
    B, C, H, W = noisy_images.shape
    if config.use_multislice:
        K = config.num_slice
        multiclass_masks_shape = (B, K, H, W)
    else:
        multiclass_masks_shape = (B, 1, H, W)

    if config.segmentation_channel_mode == "single":
        segs = convert_segbatch_to_multiclass(multiclass_masks_shape,
                                              segmentations_batch, config,
                                              device)
        noisy_images = torch.cat((noisy_images, segs), dim=1)
    elif config.segmentation_channel_mode == "multi":
        raise NotImplementedError("multi mode not implemented")
    # ---

    return noisy_images


####################
# general DDPM
####################


# --- 2.5D diff ---
def evaluate(config,
             epoch,
             pipeline,
             seg_batch=None,
             class_label_cfg=None,
             translate=False,
             use_impute=False,
             before_images=None):
    """
    Sample a batch and save each sample as .npy (no plotting, no PIL).
    Returns None.
    """
    if config.segmentation_guided:
        if not use_impute:
            images = pipeline(
                batch_size=config.eval_batch_size,
                seg_batch=seg_batch,
                class_label_cfg=class_label_cfg,
                translate=translate,
                output_type="np",  # ensure numpy output
            )  # shape [1, H, W, K]
        elif use_impute and (before_images is not None):
            images = pipeline(
                batch_size=config.eval_batch_size,
                seg_batch=seg_batch,
                class_label_cfg=class_label_cfg,
                translate=translate,
                use_impute=use_impute,
                before_images=before_images,
                output_type="np",  # ensure numpy output
            )
    else:
        images = pipeline(
            batch_size=config.eval_batch_size,
            output_type="np",
        ).images

    images = torch.as_tensor(images)
    return images

    # out_dir = os.path.join(config.output_dir, "samples")
    # os.makedirs(out_dir, exist_ok=True)
    # for i in range(images.shape[0]):
    #     # np.save(os.path.join(out_dir, f"{epoch:04d}_{i:04d}.npy"), images[i])
    #     original = seg_batch["image_filenames"][i]
    #     if not isinstance(original, str):
    #         original = str(original)

    #     prefix, axial_part = original.split("_axial_")
    #     slice_idx = axial_part.split("of")[0]

    #     fname = f"condon_{prefix}_axial_{slice_idx}.npy"
    #     np.save(os.path.join(out_dir, fname), images[i])

    # save the conditioning masks
    # if config.segmentation_guided:
    #     for seg_type, seg_tensor in seg_batch.items():
    #         if seg_type.startswith("seg_"):
    #             np.save(os.path.join(out_dir, f"{epoch:04d}_cond_{seg_type}.npy"),
    #                     seg_tensor.cpu().numpy())


# ---


# custom diffusers pipelines for sampling from segmentation-guided models
class SegGuidedDDPMPipeline(DiffusionPipeline):
    r"""
    Pipeline for segmentation-guided image generation, modified from DDPMPipeline.
    generates both-class conditioned and unconditional images if using class-conditional model without CFG, or just generates 
    conditional images guided by CFG.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
        eval_dataloader ([`torch.utils.data.DataLoader`]):
            Dataloader to load the evaluation dataset of images and their segmentations. Here only uses the segmentations to generate images.
    """
    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, eval_dataloader, external_config):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.eval_dataloader = eval_dataloader
        self.external_config = external_config  # config is already a thing

    @torch.no_grad()
    def __call__(
            self,
            batch_size: int = 1,
            generator: Optional[Union[torch.Generator,
            List[torch.Generator]]] = None,
            num_inference_steps: int = 1000,
            output_type: Optional[str] = "np",
            return_dict: bool = True,
            seg_batch: Optional[torch.Tensor] = None,
            class_label_cfg: Optional[int] = None,
            translate=False,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            seg_batch (`torch.Tensor`, *optional*, defaults to None):
                batch of segmentations to condition generation on
            class_label_cfg (`int`, *optional*, defaults to `None`):
                class label to condition generation on using CFG, if using class-conditional model

            OPTIONS FOR IMAGE TRANSLATION:
            translate (`bool`, *optional*, defaults to False):
                whether to translate images from the source domain to the target domain

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # Sample gaussian noise to begin loop
        # --- 2.5D diff ---
        if self.external_config.use_multislice:
            img_channel_ct = self.external_config.num_slice
        # ---
        else:
            if self.external_config.segmentation_channel_mode == "single":
                img_channel_ct = self.unet.config.in_channels - 1
            elif self.external_config.segmentation_channel_mode == "multi":
                img_channel_ct = self.unet.config.in_channels - len(
                    [k for k in seg_batch.keys() if k.startswith("seg_")])

        # --- 2.5D diff ---
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                img_channel_ct,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, img_channel_ct,
                           *self.unet.config.sample_size)
        # ---

        # initiate latent variable to sample from
        if not translate:
            # normal sampling; start from noise
            if self.device.type == "mps":
                # randn does not work reproducibly on mps
                image = randn_tensor(image_shape, generator=generator)
                image = image.to(self.device)
            else:
                image = randn_tensor(image_shape,
                                     generator=generator,
                                     device=self.device)
        else:
            # image translation sampling; start from source domain images, add noise up to certain step, then being there for denoising
            trans_start_t = int(self.external_config.trans_noise_level *
                                self.scheduler.config.num_train_timesteps)

            trans_start_images = seg_batch["images"]

            # Sample noise to add to the images
            noise = torch.randn(trans_start_images.shape).to(
                trans_start_images.device)
            timesteps = torch.full((trans_start_images.size(0),),
                                   trans_start_t,
                                   device=trans_start_images.device).long()
            image = self.scheduler.add_noise(trans_start_images, noise,
                                             timesteps)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            if translate:
                # if doing translation, start at chosen time step given partially-noised image
                # skip all earlier time steps (with higher t)
                if t >= trans_start_t:
                    continue

            # 1. predict noise model_output
            # first, concat segmentations to noise
            image = add_segmentations_to_noise(image, seg_batch,
                                               self.external_config,
                                               self.device)

            if self.external_config.class_conditional:
                if class_label_cfg is not None:
                    class_labels = torch.full([image.size(0)],
                                              class_label_cfg).long().to(
                        self.device)
                    model_output_cond = self.unet(
                        image, t, class_labels=class_labels).sample
                    if self.external_config.use_cfg_for_eval_conditioning:
                        # use classifier-free guidance for sampling from the given class

                        if self.external_config.cfg_maskguidance_condmodel_only:
                            image_emptymask = torch.cat(
                                (image[:, :img_channel_ct, :, :],
                                 torch.zeros_like(
                                     image[:, img_channel_ct:, :, :])),
                                dim=1)
                            model_output_uncond = self.unet(
                                image_emptymask,
                                t,
                                class_labels=torch.zeros_like(
                                    class_labels).long()).sample
                        else:
                            model_output_uncond = self.unet(
                                image,
                                t,
                                class_labels=torch.zeros_like(
                                    class_labels).long()).sample

                        # use cfg equation
                        model_output = (
                                               1. + self.external_config.cfg_weight
                                       ) * model_output_cond - self.external_config.cfg_weight * model_output_uncond
                    else:
                        # just use normal conditioning
                        model_output = model_output_cond

                else:
                    # or, just use basic network conditioning to sample from both classes
                    if self.external_config.class_conditional:
                        # if training conditionally, evaluate source domain samples
                        class_labels = torch.ones(image.size(0)).long().to(
                            self.device)
                        model_output = self.unet(
                            image, t, class_labels=class_labels).sample
            else:
                model_output = self.unet(image, t).sample
            # output is slightly denoised image

            # 2. compute previous image: x_t -> x_t-1
            # but first, we're only adding denoising the image channel (not seg channel),
            # so remove segs
            image = image[:, :img_channel_ct, :, :]
            image = self.scheduler.step(model_output,
                                        t,
                                        image,
                                        generator=generator).prev_sample

        # if training conditionally, also evaluate for target domain images
        # if not using chosen class for CFG
        if self.external_config.class_conditional and class_label_cfg is None:
            image_target_domain = randn_tensor(image_shape,
                                               generator=generator,
                                               device=self._execution_device,
                                               dtype=self.unet.dtype)

            # set step values
            self.scheduler.set_timesteps(num_inference_steps)

            for t in self.progress_bar(self.scheduler.timesteps):
                # 1. predict noise model_output
                # first, concat segmentations to noise
                # no masks in target domain so just use blank masks
                image_target_domain = torch.cat(
                    (image_target_domain,
                     torch.zeros_like(image_target_domain)),
                    dim=1)

                if self.external_config.class_conditional:
                    # if training conditionally, also evaluate unconditional model and target domain (no masks)
                    class_labels = torch.cat([
                        torch.full([image_target_domain.size(0) // 2], 2),
                        torch.zeros(image_target_domain.size(0)) // 2
                    ]).long().to(self.device)
                    model_output = self.unet(image_target_domain,
                                             t,
                                             class_labels=class_labels).sample
                else:
                    model_output = self.unet(image_target_domain, t).sample

                # 2. predict previous mean of image x_t-1 and add variance depending on eta
                # eta corresponds to η in paper and should be between [0, 1]
                # do x_t -> x_t-1
                # but first, we're only adding denoising the image channel (not seg channel),
                # so remove segs
                image_target_domain = image_target_domain[:, :
                                                             img_channel_ct, :, :]
                image_target_domain = self.scheduler.step(
                    model_output, t, image_target_domain,
                    generator=generator).prev_sample

            image = torch.cat((image, image_target_domain), dim=0)
            # will output source domain images first, then target domain images

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

# ==============modified from DDIMPipeline=================

class SegGuidedDDIMPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation, modified for seg-guided image gen.
    modified from diffusers.DDIMPipeline.
    generates both-class conditioned and unconditional images if using class-conditional model without CFG, or just generates 
    conditional images guided by CFG.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
        eval_dataloader ([`torch.utils.data.DataLoader`]):
            Dataloader to load the evaluation dataset of images and their segmentations. Here only uses the segmentations to generate images.
    
    """
    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, eval_dataloader, external_config):
        super().__init__()
        self.register_modules(unet=unet,
                              scheduler=scheduler,
                              eval_dataloader=eval_dataloader,
                              external_config=external_config)
        # ^ some reason necessary for DDIM but not DDPM.

        self.eval_dataloader = eval_dataloader
        self.external_config = external_config  # config is already a thing

        # make sure scheduler can always be converted to DDIM
        scheduler = DDIMScheduler.from_config(scheduler.config)

    @torch.no_grad()
    def __call__(
            self,
            batch_size: int = 1,
            generator: Optional[Union[torch.Generator,
            List[torch.Generator]]] = None,
            eta: float = 0.5,
            num_inference_steps: int = 50,
            use_clipped_model_output: Optional[bool] = None,
            output_type: Optional[str] = "np",
            return_dict: bool = True,
            seg_batch: Optional[torch.Tensor] = None,
            class_label_cfg: Optional[int] = None,
            translate=False,
            use_impute=False,
            before_images=None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers. A value of `0` corresponds to
                DDIM and `1` corresponds to DDPM.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            use_clipped_model_output (`bool`, *optional*, defaults to `None`):
                If `True` or `False`, see documentation for [`DDIMScheduler.step`]. If `None`, nothing is passed
                downstream to the scheduler (use `None` for schedulers which don't support this argument).
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            seg_batch (`torch.Tensor`, *optional*):
                batch of segmentations to condition generation on
            class_label_cfg (`int`, *optional*, defaults to `None`):
                class label to condition generation on using CFG, if using class-conditional model

            OPTIONS FOR IMAGE TRANSLATION:
            translate (`bool`, *optional*, defaults to False):
                whether to translate images from the source domain to the target domain

        Example:

        ```py

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        # Sample gaussian noise to begin loop
        if self.external_config.use_multislice:
            img_channel_ct = self.external_config.num_slice
        else:
            if self.external_config.segmentation_channel_mode == "single":
                img_channel_ct = self.unet.config.in_channels - 1
            elif self.external_config.segmentation_channel_mode == "multi":
                img_channel_ct = self.unet.config.in_channels - len(
                    [k for k in seg_batch.keys() if k.startswith("seg_")])

        # --- 2.5D diff ---
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                img_channel_ct,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, img_channel_ct,
                           *self.unet.config.sample_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # initiate latent variable to sample from
        if not translate:
            # normal sampling; start from noise
            image = randn_tensor(image_shape,
                                 generator=generator,
                                 device=self._execution_device,
                                 dtype=self.unet.dtype)
        else:
            # image translation sampling; start from source domain images, add noise up to certain step, then being there for denoising
            trans_start_t = int(self.external_config.trans_noise_level *
                                self.scheduler.config.num_train_timesteps)

            trans_start_images = seg_batch["images"].to(self._execution_device)

            # Sample noise to add to the images
            noise = torch.randn(trans_start_images.shape).to(
                trans_start_images.device)
            timesteps = torch.full((trans_start_images.size(0),),
                                   trans_start_t,
                                   device=trans_start_images.device).long()
            image = self.scheduler.add_noise(trans_start_images, noise,
                                             timesteps)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        # =====================GeoDiff modification=====================

        if use_impute and before_images is not None:
            eps_known = torch.randn_like(before_images)
            t_ = torch.tensor(999)

            timesteps = torch.full((image.size(0),),
                                   t_,
                                   device=image.device,
                                   dtype=torch.long)

            noisy_known = self.scheduler.add_noise(
                before_images, eps_known, timesteps)

            image[:, :-1, :, :] = noisy_known[:, 1:, :, :]  

        for t in self.progress_bar(self.scheduler.timesteps):

            if use_impute and before_images is not None and t != 999:
                noisy_img = self.scheduler.add_noise(before_images, eps_known, t)
                image[:, :-1, :, :] = noisy_img[:, 1:, :, :]

            if translate:
                if t >= trans_start_t:
                    continue

            image = torch.cat((image, seg_batch), dim=1)

            if self.external_config.class_conditional:
                if class_label_cfg is not None:
                    class_labels = torch.full([image.size(0)],
                                              class_label_cfg).long().to(
                        self.device)
                    model_output_cond = self.unet(
                        image, t, class_labels=class_labels).sample
                    if self.external_config.use_cfg_for_eval_conditioning:
                        # use classifier-free guidance for sampling from the given class
                        if self.external_config.cfg_maskguidance_condmodel_only:
                            image_emptymask = torch.cat(
                                (image[:, :img_channel_ct, :, :],
                                 torch.zeros_like(
                                     image[:, img_channel_ct:, :, :])),
                                dim=1)
                            model_output_uncond = self.unet(
                                image_emptymask,
                                t,
                                class_labels=torch.zeros_like(
                                    class_labels).long()).sample
                        else:
                            model_output_uncond = self.unet(
                                image,
                                t,
                                class_labels=torch.zeros_like(
                                    class_labels).long()).sample

                        # use cfg equation
                        model_output = (
                                               1. + self.external_config.cfg_weight
                                       ) * model_output_cond - self.external_config.cfg_weight * model_output_uncond
                    else:
                        model_output = model_output_cond

                else:
                    # or, just use basic network conditioning to sample from both classes
                    if self.external_config.class_conditional:
                        # if training conditionally, evaluate source domain samples
                        class_labels = torch.ones(image.size(0)).long().to(
                            self.device)
                        model_output = self.unet(
                            image, t, class_labels=class_labels).sample
            else:
                model_output = self.unet(image, t).sample

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            # but first, we're only adding denoising the image channel (not seg channel),
            # so remove segs
            image = image[:, :img_channel_ct, :, :]
            image = self.scheduler.step(
                model_output,
                t,
                image,
                eta=eta,
                use_clipped_model_output=use_clipped_model_output,
                generator=generator).prev_sample

        # if training conditionally, also evaluate for target domain images
        # if not using chosen class for CFG
        if self.external_config.class_conditional and class_label_cfg is None:
            image_target_domain = randn_tensor(image_shape,
                                               generator=generator,
                                               device=self._execution_device,
                                               dtype=self.unet.dtype)

            # set step values
            self.scheduler.set_timesteps(num_inference_steps)

            for t in self.progress_bar(self.scheduler.timesteps):
                # 1. predict noise model_output
                # first, concat segmentations to noise
                # no masks in target domain so just use blank masks
                image_target_domain = torch.cat(
                    (image_target_domain,
                     torch.zeros_like(image_target_domain)),
                    dim=1)

                if self.external_config.class_conditional:
                    # if training conditionally, also evaluate unconditional model and target domain (no masks)
                    class_labels = torch.cat([
                        torch.full([image_target_domain.size(0) // 2], 2),
                        torch.zeros(image_target_domain.size(0) // 2)
                    ]).long().to(self.device)
                    model_output = self.unet(image_target_domain,
                                             t,
                                             class_labels=class_labels).sample
                else:
                    model_output = self.unet(image_target_domain, t).sample

                # 2. predict previous mean of image x_t-1 and add variance depending on eta
                # eta corresponds to η in paper and should be between [0, 1]
                # do x_t -> x_t-1
                # but first, we're only adding denoising the image channel (not seg channel),
                # so remove segs
                image_target_domain = image_target_domain[:, :
                                                             img_channel_ct, :, :]
                image_target_domain = self.scheduler.step(
                    model_output,
                    t,
                    image_target_domain,
                    eta=eta,
                    use_clipped_model_output=use_clipped_model_output,
                    generator=generator).prev_sample

            image = torch.cat((image, image_target_domain), dim=0)
            # will output source domain images first, then target domain images

        # image = (image / 2 + 0.5).clamp(0, 1)
        # image = image.cpu().permute(0, 2, 3, 1).numpy()
        return image
        # if output_type == "pil":
        #     image = self.numpy_to_pil(image)

        # if not return_dict:
        #     return (image, )

        # return ImagePipelineOutput(images=image)


# def evaluate_geo_generation(config,
#                             model,
#                             noise_scheduler,
#                             eval_dataloader,
#                             class_label_cfg=None,
#                             translate=False,
#                             eval_mask_removal=False,
#                             eval_blank_mask=False,
#                             device='cuda'):
#     """
#     The first and important version
#     general function to evaluate (possibly mask-guided) trained image generation model in useful ways.
#     also has option to use CFG for class-conditioned sampling (otherwise, class-conditional models will be evaluated using naive class conditioning and sampling from both classes).
#
#     can also evaluate for image translation.
#     """
#
#     # for loading segs to condition on:
#     eval_dataloader2 = iter(eval_dataloader)
#
#     if config.segmentation_guided:
#         seg_batch = next(eval_dataloader2)  # [1*256*256*256, 1*256*256*256]
#         if eval_blank_mask:
#             # use blank masks
#             for k, v in seg_batch.items():
#                 if k.startswith("seg_"):
#                     seg_batch[k] = torch.zeros_like(v)
#
#     # setup for sampling
#     # After each epoch you optionally sample some demo images with evaluate() and save the model
#     if config.model_type == "DDPM":
#         if config.segmentation_guided:
#             pipeline = SegGuidedDDPMPipeline(unet=model.module,
#                                              scheduler=noise_scheduler,
#                                              eval_dataloader=eval_dataloader,
#                                              external_config=config)
#         else:
#             pipeline = diffusers.DDPMPipeline(unet=model.module,
#                                               scheduler=noise_scheduler)
#     elif config.model_type == "DDIM":
#         if config.segmentation_guided:
#             pipeline = SegGuidedDDIMPipeline(unet=model.module,
#                                              scheduler=noise_scheduler,
#                                              eval_dataloader=eval_dataloader,
#                                              external_config=config)
#         else:
#             pipeline = diffusers.DDIMPipeline(unet=model.module,
#                                               scheduler=noise_scheduler)
#
#     # --- 2.5D diff ---
#     if config.segmentation_guided:
#         seg_label = seg_batch[1]
#
#         final_volume = torch.randn_like(seg_label).to(device)
#         seg_label = seg_label.to(device)
#
#         for j in range(seg_label.shape[1] - 8):
#             seg_subvol = seg_label[:, j:j + 8, :, :]
#
#             if j == 0:
#                 before_images = pipeline(
#                     batch_size=config.eval_batch_size,
#                     seg_batch=seg_subvol,
#                     class_label_cfg=class_label_cfg,
#                     translate=translate,
#                     output_type="np",
#                 )  # shape: [B, H, W, K]
#                 final_volume[:, j:j + 8, :, :] = before_images
#             else:
#                 before_images = pipeline(batch_size=config.eval_batch_size,
#                                          seg_batch=seg_subvol,
#                                          class_label_cfg=class_label_cfg,
#                                          translate=translate,
#                                          output_type="np",
#                                          use_impute=True,
#                                          before_images=before_images)
#
#                 final_volume[:, j + 7, :, :] = before_images[:, -1, :, :]
#
#
#     else:
#         images = pipeline(
#             batch_size=config.eval_batch_size,
#             output_type="np",
#         ).images
#
#     # Save each sample as .npy (no plotting)
#     # final_volume = (final_volume / 2 + 0.5).clip(0, 1).cpu().numpy()
#     final_volume = ((final_volume - final_volume.min()) / (final_volume.max() - final_volume.min())).cpu().numpy()
#     test_dir = os.path.join(config.results_output_dir, "samples")
#     os.makedirs(test_dir, exist_ok=True)
#     np.save(os.path.join(test_dir, "images.npy"), final_volume)
#     np.save(os.path.join(test_dir, "labels.npy"), seg_label)
#     # ---
#
#     return

def evaluate_geo_generation(config,
                            model,
                            noise_scheduler,
                            eval_dataloader,
                            class_label_cfg=None,
                            translate=False,
                            eval_mask_removal=False,
                            eval_blank_mask=False,
                            device='cuda'):
    """
    This function evaluates the 2.5D diffusion model by using segmentation voxelized from the HUG-VAS output.
    """

    # for loading segs to condition on:
    # eval_dataloader2 = iter(eval_dataloader)

    for seg_batch in eval_dataloader:

        if config.segmentation_guided:
            # seg_batch = next(eval_dataloader2)  # [1*256*256*256, 1*256*256*256]
            if eval_blank_mask:
                # use blank masks
                for k, v in seg_batch.items():
                    if k.startswith("seg_"):
                        seg_batch[k] = torch.zeros_like(v)

        # setup for sampling
        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if config.model_type == "DDPM":
            if config.segmentation_guided:
                pipeline = SegGuidedDDPMPipeline(unet=model.module,
                                                 scheduler=noise_scheduler,
                                                 eval_dataloader=eval_dataloader,
                                                 external_config=config)
            else:
                pipeline = diffusers.DDPMPipeline(unet=model.module,
                                                  scheduler=noise_scheduler)
        elif config.model_type == "DDIM":
            if config.segmentation_guided:
                pipeline = SegGuidedDDIMPipeline(unet=model.module,
                                                 scheduler=noise_scheduler,
                                                 eval_dataloader=eval_dataloader,
                                                 external_config=config)
            else:
                pipeline = diffusers.DDIMPipeline(unet=model.module,
                                                  scheduler=noise_scheduler)

        # --- 2.5D diff ---
        if config.segmentation_guided:
            seg_label = seg_batch[1]

            final_volume = torch.randn_like(seg_label,dtype=torch.float32).to(device)
            seg_label = seg_label.to(device)

            for j in range(seg_label.shape[1] - 8):
                seg_subvol = seg_label[:, j:j + 8, :, :]

                if j == 0:
                    before_images = pipeline(
                        batch_size=config.eval_batch_size,
                        seg_batch=seg_subvol,
                        class_label_cfg=class_label_cfg,
                        translate=translate,
                        output_type="np",
                    )  # shape: [B, H, W, K]
                    final_volume[:, j:j + 8, :, :] = before_images
                else:
                    before_images = pipeline(batch_size=config.eval_batch_size,
                                             seg_batch=seg_subvol,
                                             class_label_cfg=class_label_cfg,
                                             translate=translate,
                                             output_type="np",
                                             use_impute=True,
                                             before_images=before_images)

                    final_volume[:, j + 7, :, :] = before_images[:, -1, :, :]


        else:
            images = pipeline(
                batch_size=config.eval_batch_size,
                output_type="np",
            ).images

        # Save each sample as .npy (no plotting)
        # final_volume = (final_volume / 2 + 0.5).clip(0, 1).cpu().numpy()
        final_volume = ((final_volume - final_volume.min()) / (final_volume.max() - final_volume.min())).cpu().numpy()
        test_dir = os.path.join(config.results_output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        img_dir = os.path.join(test_dir, "img")
        os.makedirs(img_dir, exist_ok=True)
        label_dir = os.path.join(test_dir, "label")
        os.makedirs(label_dir, exist_ok=True)
        case_name = pathlib.Path(seg_batch[0][0])
        np.save(os.path.join(img_dir, f"{case_name.stem}.npy"), final_volume)
        # np.save(os.path.join(label_dir, f"{case_name}.npy"), seg_label)
        shutil.copy(case_name, os.path.join(label_dir, f"{case_name.stem}.nrrd"))
        # ---

    return
