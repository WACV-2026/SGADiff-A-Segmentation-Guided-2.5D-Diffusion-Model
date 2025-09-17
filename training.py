from tqdm import tqdm
import numpy as np
import os
import logging
import json
from dataclasses import asdict, is_dataclass

import torch
from torch import nn
import torch.nn.functional as F

import diffusers

from eval import evaluate, add_segmentations_to_noise, SegGuidedDDPMPipeline, SegGuidedDDIMPipeline
from datetime import datetime


# --- main train loop ----------------------------------------------------------
def train_loop(config,
               model,
               noise_scheduler,
               optimizer,
               train_dataloader,
               eval_dataloader,
               lr_scheduler,
               accelerator,
               device: str = "cuda"):
    """
    This function is modified from the Seg-Diff repository
    """

    # ----------------------- 1. bookkeeping -----------------------------------
    run_name = f"{config.model_type.lower()}-{config.dataset}-{config.image_size}"
    if config.segmentation_guided:
        run_name += "-segguided"

    dt_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    output_dir = os.path.join(config.output_dir, f"{run_name}-{dt_str}")
    os.makedirs(output_dir, exist_ok=True)
    config.output_dir = output_dir

    log_file = os.path.join(output_dir, "training.log")

    cfg_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(cfg_path):  # don’t overwrite on resume
        cfg_dict = asdict(config) if is_dataclass(config) else vars(config)
        with open(cfg_path, "w") as f:
            json.dump(cfg_dict, f, indent=2, default=str)

    logging.basicConfig(
        filename=log_file,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(run_name)

    logger.info("======== New run started ========")
    logger.info(f"config: {config}")

    # ----------------------- 2. training --------------------------------------
    global_step = 0
    start_epoch = config.resume_epoch or 0
    # eval_dataloader = iter(eval_dataloader)  # keep for evaluation batches

    for epoch in range(start_epoch, config.num_epochs):
        logger.info(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"Epoch {epoch + 1}/{config.num_epochs}")

        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        model.train()

        for step, batch in enumerate(train_dataloader):
            clean_images = batch[0].to(device)

            # Sample noise to add to the images
            noise = torch.randn_like(clean_images)
            bs = clean_images.size(0)

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps, (bs, ),
                device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (FORWARD DIFFUSION PROCESS)
            noisy_images = noise_scheduler.add_noise(clean_images, noise,
                                                     timesteps)

            if config.segmentation_guided:
                # noisy_images = add_segmentations_to_noise(
                #     noisy_images, batch, config, device)
                noisy_images = torch.cat((noisy_images, batch[1].to(device)),
                                         dim=1)

            # --------------- forward & loss ----------------
            # Predict the noise residual
            if config.class_conditional:
                class_labels = torch.ones(bs, dtype=torch.long, device=device)
                # classifier-free guidance
                if np.random.uniform() <= config.cfg_p_uncond:
                    class_labels.zero_()

                noise_pred = model(noisy_images,
                                   timesteps,
                                   class_labels=class_labels,
                                   return_dict=False)[0]
            else:
                noise_pred = model(noisy_images, timesteps,
                                   return_dict=False)[0]

            loss = F.mse_loss(noise_pred, noise)
            # loss.backward()
            accelerator.backward(loss)

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # -------- extra pass for target‑domain images ----------
            if config.class_conditional:
                tgt_imgs = batch["images_target"].to(device)
                tgt_noise = torch.randn_like(tgt_imgs)
                tgt_ts = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (tgt_imgs.size(0), ),
                    device=device).long()

                tgt_noisy = noise_scheduler.add_noise(tgt_imgs, tgt_noise,
                                                      tgt_ts)

                if config.segmentation_guided:
                    tgt_noisy = torch.cat(
                        (tgt_noisy, torch.zeros_like(tgt_noisy)), dim=1)

                tgt_labels = torch.full((tgt_noisy.size(0), ),
                                        2,
                                        dtype=torch.long,
                                        device=device)
                if np.random.uniform() <= config.cfg_p_uncond:
                    tgt_labels.zero_()

                tgt_pred = model(tgt_noisy,
                                 tgt_ts,
                                 class_labels=tgt_labels,
                                 return_dict=False)[0]
                loss_tgt = F.mse_loss(tgt_pred, tgt_noise)
                loss_tgt.backward()

                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # --------------- logging ----------------------
            lr = lr_scheduler.get_last_lr()[0]
            if config.class_conditional:
                logger.info(f"step={global_step:>7} "
                            f"loss={loss.item():.6f} "
                            f"loss_tgt={loss_tgt.item():.6f} "
                            f"lr={lr:.6e}")
            else:
                logger.info(f"step={global_step:>7} "
                            f"loss={loss.item():.6f} "
                            f"lr={lr:.6e}")

            progress_bar.set_postfix(loss=loss.item(),
                                     **({
                                         "loss_target_domain": loss_tgt.item()
                                     } if config.class_conditional else {}),
                                     lr=lr)
            global_step += 1
            progress_bar.update(1)

        # ---------------- evaluation / checkpointing ---------------
        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if config.model_type == "DDPM":
            if config.segmentation_guided:
                pipeline = SegGuidedDDPMPipeline(
                    unet=model.module,
                    scheduler=noise_scheduler,
                    eval_dataloader=eval_dataloader,
                    external_config=config)
            else:
                if config.class_conditional:
                    raise NotImplementedError(
                        "TODO: Conditional training not implemented for non-seg-guided DDPM"
                    )
                else:
                    pipeline = diffusers.DDPMPipeline(
                        unet=model.module, scheduler=noise_scheduler)
        elif config.model_type == "DDIM":
            if config.segmentation_guided:
                pipeline = SegGuidedDDIMPipeline(
                    unet=model.module,
                    scheduler=noise_scheduler,
                    eval_dataloader=eval_dataloader,
                    external_config=config)
            else:
                if config.class_conditional:
                    raise NotImplementedError(
                        "TODO: Conditional training not implemented for non-seg-guided DDIM"
                    )
                else:
                    pipeline = diffusers.DDIMPipeline(
                        unet=model.module, scheduler=noise_scheduler)

        model.eval()

        if (
                epoch + 1
        ) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            if config.segmentation_guided:
                eval_dataloader2 = iter(eval_dataloader)
                seg_batch = next(eval_dataloader2)

                # for multi-slice
                seg_original_img, seg_label = seg_batch
                seg_original_img = seg_original_img.to(device)
                seg_label = seg_label.to(device)

                seg_label = seg_label.squeeze()  # (D, H, W)
                seg_label = seg_label.unsqueeze(0)  # (1, D, H, W)
                assert len(seg_label.shape
                           ) == 4, f"seg_label shape: {seg_label.shape}"

                final_volume = torch.randn_like(seg_label).to(device)

                for i in range(seg_label.shape[1] - 8):
                    if i == 0:
                        before_images = evaluate(config, epoch, pipeline,
                                                 seg_label[:, i:i + 8, :, :])
                        before_images = before_images.squeeze().unsqueeze(0)
                    else:
                        before_images = evaluate(config,
                                                 epoch,
                                                 pipeline,
                                                 seg_label[:, i:i + 8, :, :],
                                                 use_impute=True,
                                                 before_images=before_images)
                        before_images = before_images.squeeze().unsqueeze(0)

                    final_volume[:, i:i + 8, :, :] = before_images

            else:
                evaluate(config, epoch, pipeline)

            os.makedirs(os.path.join(config.output_dir, "samples"),
                        exist_ok=True)
            final_volume = final_volume.squeeze().cpu().numpy()
            np.save(
                os.path.join(config.output_dir, "samples",
                             f"epoch{epoch+1:03d}_final_volume.npy"),
                final_volume)

        if (
                epoch + 1
        ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            # pipeline.save_pretrained(config.output_dir)
            ckpt_dir = os.path.join(config.output_dir,
                                    f"checkpoint-epoch{epoch+1}")
            pipeline.save_pretrained(ckpt_dir)

    logger.info("======== Training finished ========")
