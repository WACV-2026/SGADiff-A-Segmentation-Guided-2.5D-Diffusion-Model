import os

import torch

import monai
from monai.config import print_config

import diffusers
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch.nn as nn

# custom imports
from utils_diff import TrainConfig
from dataset import NrrdDataset, NRRDDataset, InfDataset
from training import train_loop
from eval import evaluate_generation, evaluate_sample_many, evaluate_geo_generation

from monai.data import DataLoader
from accelerate import Accelerator


def main():
    num_workers = 0

    ### Evaluation configuration
    cfg = TrainConfig(
        mode="eval",
        model_type="DDIM",
        output_dir=
        # "runs/ddim-AVT_testing-256-segguided-20250817-221307/checkpoint-epoch1000",
        "runs/ddim-AVT_testing-256-segguided-20250817-221307/checkpoint-epoch510",
        dataset="AVT",
        image_size=256,
        eval_batch_size=1,
        eval_sample_size=900,

        # segmentation guided options
        segmentation_guided=True,
        segmentation_channel_mode="single",
        num_segmentation_classes=2,
        # seg_dir="../data/seg/test",
        # seg_dir="../data/mesh_ct/seg",  # change this to the seg dir
        # seg_dir="../data/data_filtered/data_filtered/mesh_seg",  # change this to the seg dir
        # seg_dir="../data/seg/select_test",  # change this to the seg dir
        seg_dir="../data/selected_new_seg",  # change this to the seg dir
        # seg_dir="../data/mesh_disease_seg",  # change this to the seg dir
        img_dir="../data/img/test",

        # 2.5D options
        use_multislice=True,
        num_slice=8,
        eval_blank_mask=False,
        results_output_dir="../data/paper_results",
    )

    # os.makedirs(cfg.output_dir, exist_ok=True)

    # --- Load Dataset ---------------------------------------------
    if cfg.use_multislice:
        # use MultiSliceDataset for 2.5D diffusion
        # train_ds = NrrdDataset(cfg.img_dir,
        #                        cfg.seg_dir,
        #                        mode="train",
        #                        seq_len=cfg.num_slice)
        # val_ds = NrrdDataset(cfg.img_dir, cfg.seg_dir, mode="test")
        val_ds = InfDataset(cfg.seg_dir, mode="test")
    else:
        # use NRRDDataset for 2D diffusion
        train_ds = NRRDDataset(cfg.img_dir,
                               cfg.seg_dir,
                               split="train",
                               img_size=cfg.image_size,
                               segmentation_guided=cfg.segmentation_guided)
        val_ds = NRRDDataset(cfg.img_dir,
                             cfg.seg_dir,
                             split="val",
                             img_size=cfg.image_size,
                             segmentation_guided=cfg.segmentation_guided)

    # train_loader = DataLoader(train_ds,
    #                           batch_size=cfg.train_batch_size,
    #                           shuffle=True,
    #                           num_workers=num_workers,
    #                           drop_last=True)
    train_loader = None  # only for eval
    val_loader = DataLoader(val_ds,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            drop_last=False)

    # batch = next(iter(train_loader))
    # print("Batch tensor keys :", batch.keys())
    # print("Batch 'images'    :", batch["images"].shape)      # (B, 1, 256, 256)

    # --- Define Model, Optimizer, Scheduler -----------------------------
    in_channels = cfg.num_img_channels * cfg.num_slice  # for 2.5D, we have K slices
    if cfg.segmentation_guided:
        assert cfg.num_segmentation_classes is not None
        assert cfg.num_segmentation_classes > 1, "must have at least 2 segmentation classes (INCLUDING background)"
        if cfg.segmentation_channel_mode == "single":
            if cfg.use_multislice:
                in_channels = cfg.num_slice + cfg.num_slice  # for 2.5D, we have K slices + K slices for segmentation
            else:
                in_channels += 1
        elif cfg.segmentation_channel_mode == "multi":
            if cfg.use_multislice:
                raise NotImplementedError(
                    "multi-channel segmentation not implemented for 2.5D diffusion"
                )
            else:
                in_channels = len(os.listdir(cfg.seg_dir)) + in_channels

    model = diffusers.UNet2DModel(
        sample_size=cfg.image_size,  # the target image resolution
        in_channels=
        in_channels,  # the number of input channels, 3 for RGB images
        out_channels=cfg.num_img_channels *
                     cfg.num_slice,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(
            128, 128, 256, 256, 512,
            512),  # the number of output channes for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D"),
    )

    if (cfg.mode == "train"
        and cfg.resume_epoch is not None) or "eval" in cfg.mode:
        if cfg.mode == "train":
            print("resuming from model at training epoch {}".format(
                cfg.resume_epoch))
        elif "eval" in cfg.mode:
            print("loading saved model...")
        model = model.from_pretrained(os.path.join(cfg.output_dir, 'unet'),
                                      use_safetensors=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    accelerator = Accelerator()
    # model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    # device = accelerator.device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)

    model = nn.DataParallel(model)
    model.to(device)

    if cfg.model_type == "DDPM":
        noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=1000)
    elif cfg.model_type == "DDIM":
        noise_scheduler = diffusers.DDIMScheduler(num_train_timesteps=1000)

    # --- Training ----------------------------------------------------
    if cfg.mode == "train":
        # training setup

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=cfg.lr_warmup_steps,
            num_training_steps=(len(train_loader) * cfg.num_epochs),
        )

        # train
        train_loop(cfg,
                   model,
                   noise_scheduler,
                   optimizer,
                   train_loader,
                   val_loader,
                   lr_scheduler,
                   accelerator=accelerator,
                   device=device)
    elif cfg.mode == "eval":
        '''
        This branch evaluates the 2.5D diffusion model for SGADiff
        '''
        evaluate_geo_generation(cfg,
                                model,
                                noise_scheduler,
                                val_loader,
                                eval_mask_removal=cfg.eval_mask_removal,
                                eval_blank_mask=cfg.eval_blank_mask,
                                device=device)

    elif cfg.mode == "eval_many":
        """
        generate many images and save them to a directory, saved individually
        """
        evaluate_sample_many(cfg.eval_sample_size,
                             cfg,
                             model,
                             noise_scheduler,
                             val_loader,
                             device=device)

    else:
        raise ValueError("mode \"{}\" not supported.".format(cfg.mode))


if __name__ == "__main__":
    main()
