from PIL import Image
from dataclasses import dataclass, asdict

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid


#--- Config Class ---
@dataclass
class TrainConfig:
    mode: str ="train"
    model_type: str = "DDPM"
    image_size: int = 256  # the generated image resolution
    num_img_channels: int = 1
    train_batch_size: int = 32
    eval_batch_size: int = 8  # how many images to sample during evaluation
    num_epochs: int = 200
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 20
    save_model_epochs: int = 30
    mixed_precision: str = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir: str = None
    img_dir: str = None
    seg_dir: str = None

    # push_to_hub: bool = False  # whether to upload the saved model to the HF Hub
    # hub_private_repo: bool = False
    # overwrite_output_dir: bool = True  # overwrite the old model when re-running the notebook
    seed: int = 0

    # custom options
    segmentation_guided: bool = True
    segmentation_channel_mode: str = "single"
    num_segmentation_classes: int = 2 # INCLUDING background
    use_ablated_segmentations: bool = False
    dataset: str = "AVT_dongyang"
    resume_epoch: int = None

    eval_sample_size: int = 100
    eval_mask_removal: bool = True
    eval_blank_mask: bool = True

    # 2.5D options
    use_multislice: bool = False
    num_slice: int = 1 # number of slices to use in 2.5D diffusion, 
                       # 1 for 2D diffusion, >1 for 2.5D diffusion
    stride: int = 1

    #  EXPERIMENTAL/UNTESTED: classifier-free class guidance and image translation
    class_conditional: bool = False
    cfg_p_uncond: float = 0.2 # p_uncond in classifier-free guidance paper
    cfg_weight: float = 0.3 # w in the paper
    trans_noise_level: float = 0.5 # ratio of time step t to noise trans_start_images to total T before denoising in translation. e.g. value of 0.5 means t = 500 for default T = 1000.
    use_cfg_for_eval_conditioning: bool = True  # whether to use classifier-free guidance for or just naive class conditioning for main sampling loop
    cfg_maskguidance_condmodel_only: bool = True  # if using mask guidance AND cfg, only give mask to conditional network
    # ^ this is because giving mask to both uncond and cond model make class guidance not work 
    # (see "Classifier-free guidance resolution weighting." in ControlNet paper)

    results_output_dir: str = None  # where to save the evaluation results