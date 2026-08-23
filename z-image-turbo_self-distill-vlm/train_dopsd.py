import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DeepSpeedPlugin, DistributedType
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.utils.deepspeed import get_active_deepspeed_plugin
from accelerate.logging import get_logger
from diffusers import ZImagePipeline
from diffusers.utils.torch_utils import is_compiled_module
import tqdm
import logging
from pathlib import Path
import json
import sys
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import math
from torchvision.utils import make_grid
from dataset import TextImageDataset, AspectBatchSampler, CustomDataLoader, parse_ratios
from dataset_validate import TextPromptDataset
from dopsd_extension import (
    DStepObserverSession,
    DopsdExtensionContext,
    DopsdTrainingExtension,
    observed_clip_branch,
    require_d_step_observer,
)
from local_paths import resolve_existing_path
from PIL import Image
from arguments import parse_args
from utils import _encode_prompt, create_generator
from ema_utils import *
from vlm_utils import load_matching_state_dict,get_qwen3vl_zimage_prompt_embeds

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from efficient_fewstep.methods.dopsd.diagnostics import (
    DopsdDiagnosticsConfig,
    DopsdDiagnosticsRecorder,
)

logger = get_logger(__name__)

def array2grid(x):
    n_images = x.size(0)
    height = x.size(2)
    width = x.size(3)
    aspect_ratio = width / height
    nrow = max(1, round(math.sqrt(n_images / aspect_ratio)))
    grid = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return grid


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def parse_teacher_timestep_indices(raw_indices, num_training_steps):
    if raw_indices is None or str(raw_indices).strip().lower() in {"", "all"}:
        return set(range(num_training_steps))

    indices = set()
    for raw_part in str(raw_indices).split(","):
        part = raw_part.strip()
        if not part:
            continue
        index = int(part)
        if index < 0 or index >= num_training_steps:
            raise ValueError(
                f"teacher timestep index {index} is outside [0, {num_training_steps - 1}]"
            )
        indices.add(index)

    if not indices:
        raise ValueError("--teacher-timestep-indices must select at least one timestep")
    return indices


def default_training_timesteps(num_training_steps):
    if num_training_steps == 4:
        return [0, 100.0000014901161, 250, 500]
    if num_training_steps == 8:
        timesteps = [
            1000.0000,
            976.8991,
            947.7647,
            909.8782,
            858.5987,
            785.2998,
            671.9212,
            473.2203,
        ]
        return [1000 - t for t in timesteps]
    raise NotImplementedError


def parse_training_timesteps(raw_timesteps, num_training_steps):
    if raw_timesteps is None or str(raw_timesteps).strip() == "":
        return default_training_timesteps(num_training_steps)

    timesteps = []
    for raw_part in str(raw_timesteps).split(","):
        part = raw_part.strip()
        if not part:
            continue
        timestep = float(part)
        if timestep < 0 or timestep >= 1000:
            raise ValueError(f"training timestep {timestep} is outside [0, 1000)")
        timesteps.append(timestep)

    if len(timesteps) != num_training_steps:
        raise ValueError(
            f"--training-timesteps has {len(timesteps)} values, "
            f"but --num-training-steps is {num_training_steps}"
        )
    if any(curr <= prev for prev, curr in zip(timesteps, timesteps[1:])):
        raise ValueError("--training-timesteps must be strictly increasing")
    return timesteps


def select_adaptive_teacher_timesteps(candidate_indices, scores, top_k):
    if top_k <= 0 or top_k >= len(candidate_indices):
        return set(candidate_indices)

    def score_key(index):
        score = scores.get(index)
        return float("-inf") if score is None else float(score)

    selected = sorted(candidate_indices, key=score_key, reverse=True)[:top_k]
    return set(selected)


def active_teacher_timestep_indices(args, optimizer_step, base_indices, warmup_indices, adaptive_scores):
    if args.teacher_timestep_adaptive_top_k > 0:
        adaptive_warmup_steps = int(args.teacher_timestep_adaptive_warmup_steps)
        if optimizer_step <= adaptive_warmup_steps:
            return set(base_indices)
        return select_adaptive_teacher_timesteps(
            sorted(base_indices),
            adaptive_scores,
            int(args.teacher_timestep_adaptive_top_k),
        )

    if int(args.teacher_timestep_warmup_steps) > 0 and optimizer_step <= int(args.teacher_timestep_warmup_steps):
        return set(warmup_indices)
    return set(base_indices)


def update_adaptive_teacher_score(args, scores, back_step, metric_value):
    if args.teacher_timestep_adaptive_top_k <= 0 or metric_value is None:
        return
    value = float(metric_value.detach().float().mean().item())
    previous = scores.get(back_step)
    if previous is None:
        scores[back_step] = value
    else:
        ema = float(args.teacher_timestep_adaptive_ema)
        scores[back_step] = ema * float(previous) + (1.0 - ema) * value



@torch.no_grad()
def decode_latents_to_images(latents, pipeline):
    latents = latents.to(device=pipeline.vae.device, dtype=pipeline.vae.dtype)
    latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    images = pipeline.vae.decode(latents, return_dict=False)[0]
    images = (images / 2 + 0.5).clamp(0, 1)
    return images


def save_student_teacher_trajectory(pipeline, student_x0_traj, teacher_x0_traj, save_dir, global_step, accelerator, max_size=None):
    import os, math, numpy as np
    from PIL import Image, ImageDraw, ImageFont
    os.makedirs(save_dir, exist_ok=True)

    def to_uint8(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        if x.ndim == 4 and x.shape[1] in (1, 3):
            x = np.transpose(x, (0, 2, 3, 1))
        return x if x.dtype == np.uint8 else (np.clip(x, 0, 1) * 255).round().astype(np.uint8)

    def grid(x, nrow=4, pad=2, bg=255):
        assert x.ndim == 4, f"grid expects 4D input, got {x.shape}"
        n, h, w, c = x.shape
        nrow = max(1, min(nrow, n))
        ncol = math.ceil(n / nrow)
        g = np.full((ncol * h + pad * (ncol - 1), nrow * w + pad * (nrow - 1), c), bg, np.uint8)
        for k in range(n):
            r, col = divmod(k, nrow)
            y, z = r * (h + pad), col * (w + pad)
            g[y:y + h, z:z + w] = x[k]
        return g

    def add_titles_and_concat(a, b, pad=16, title_h=36, bg=255):
        h, w1, c = a.shape
        _, w2, _ = b.shape
        canvas = np.full((title_h + max(a.shape[0], b.shape[0]), w1 + pad + w2, c), bg, np.uint8)
        canvas[title_h:title_h + a.shape[0], :w1] = a
        canvas[title_h:title_h + b.shape[0], w1 + pad:w1 + pad + w2] = b
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.text((10, 10), "Student", fill=(0, 0, 0), font=font)
        draw.text((w1 + pad + 10, 10), "Teacher", fill=(0, 0, 0), font=font)
        return img

    for i, (sx0, tx0) in enumerate(zip(student_x0_traj, teacher_x0_traj)):
        s = decode_latents_to_images(sx0[:4], pipeline).float()
        t = decode_latents_to_images(tx0[:4], pipeline).float()

        if accelerator.is_main_process:

            t_dir = os.path.join(save_dir, f"t{i}")
            os.makedirs(t_dir, exist_ok=True)

            s_np = to_uint8(s)
            t_np = to_uint8(t)

            single_img_dir = f"{t_dir}/one_img"
            os.makedirs(single_img_dir, exist_ok=True)
            Image.fromarray(s_np[0]).save(os.path.join(single_img_dir, f"step_{global_step}_student_single.png"))
            Image.fromarray(t_np[0]).save(os.path.join(single_img_dir, f"step_{global_step}_teacher_single.png"))

            nrow = 4 if s_np.shape[1] <= 1024 else 2
            img = add_titles_and_concat(grid(s_np, nrow=nrow), grid(t_np, nrow=nrow))

            if max_size is not None:
                w, h = img.size
                scale = min(max_size[0] / w, max_size[1] / h, 1.0)
                if scale < 1:
                    img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

            img.save(os.path.join(t_dir, f"step_{global_step}_student_teacher_x0.png"))


#################################################################################
#                                  Training Loop                                #
#################################################################################

def run_training(args, extension=None):
    extension = extension or DopsdTrainingExtension()
    d_step_observer = require_d_step_observer(extension.d_step_observer())
    teacher_conditioning_mode = extension.teacher_conditioning_mode()
    if teacher_conditioning_mode not in {"multimodal", "student_text"}:
        raise ValueError(
            "teacher conditioning mode must be multimodal or student_text"
        )
    teacher_adapter_update_mode = extension.teacher_adapter_update_mode()
    if teacher_adapter_update_mode not in {"ema", "frozen"}:
        raise ValueError("teacher adapter update mode must be ema or frozen")
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    ds_config = args.deepspeed_config

    zero2_plugin_a = DeepSpeedPlugin(hf_ds_config=ds_config)
    deepspeed_plugins = {"z2_a": zero2_plugin_a}

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        deepspeed_plugins=deepspeed_plugins,
    )

    os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    save_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)

    if accelerator.is_main_process:
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)

        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")

    diagnostics = DopsdDiagnosticsRecorder(
        DopsdDiagnosticsConfig.from_args(args, save_dir),
        accelerator=accelerator,
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Create pipe :
    pipeline = ZImagePipeline.from_pretrained(
        args.pretrained_model,
        low_cpu_mem_usage=False,
    )

    num_channels_latents = pipeline.transformer.in_channels

    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(args.use_lora <= 1)
    tokenizer = pipeline.tokenizer


    # The historical path is unchanged.  Text-identical extensions have no
    # use for Qwen and, importantly, do not pay its memory/load cost.
    vl_model = None
    processor = None
    if teacher_conditioning_mode == "multimodal":
        vl_model_name = args.teacher_vlm_model_path
        min_pixels = 512 * 512
        max_pixels = 768 * 768
        from transformers import AutoProcessor, AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(vl_model_name, min_pixels=min_pixels, max_pixels=max_pixels)
        vl_model = AutoModelForImageTextToText.from_pretrained(
            vl_model_name,
        )
        missing_keys, unexpected_keys = load_matching_state_dict(
            target_module=vl_model.model.language_model,
            source_state_dict=pipeline.text_encoder.state_dict(),
            verbose=False,
        )
        vl_model.requires_grad_(False)
        vl_model.to(accelerator.device, dtype=inference_dtype)
        vl_model.eval()

        if accelerator.is_main_process:
            logger.info(f"Teacher VLM loaded: {vl_model_name}, dtype: {vl_model.parameters().__next__().dtype}")
    elif accelerator.is_main_process:
        logger.info("Teacher uses the student's exact text embeddings; Qwen3-VL is not loaded")




    # disable progress bar for cold start
    pipeline.set_progress_bar_config(disable=True)

    # init lora
    if args.use_lora > 1:
        # Set correct lora layers
        target_modules = [
            "feed_forward.w1",
            "feed_forward.w2",
            "feed_forward.w3",
            "attention.to_k",
            "attention.to_q",
            "attention.to_v",
            "attention.to_out.0",
        ]
        pipeline.transformer = init_dual_lora_transformer(
            transformer=pipeline.transformer,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            current_adapter_name="student",
            old_adapter_name="teacher",
            old_init_from_current=True,
        )
        extension.initialize_adapter_state(pipeline.transformer)
        extension.validate_teacher_adapter_state(
            pipeline.transformer,
            optimizer_step=0,
            event="initialized",
        )

    # we use ema in full-finetune
    else:
        raise NotImplementedError("Full finetuning is not implemented here, please set --use-lora to > 1 for now.")

    # Move vae and text_encoder to device and cast to inference_dtype
    if args.vae_dtype == "fp32":
        vae_dtype = torch.float32
        pipeline.vae.to(accelerator.device, dtype=vae_dtype)
    else:
        vae_dtype = inference_dtype
        pipeline.vae.to(accelerator.device, dtype=vae_dtype)
    # avoid OOM
    pipeline.vae.enable_slicing()
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)

    gen_model = pipeline.transformer
    gen_model_trainable_parameters = list(filter(lambda p: p.requires_grad, gen_model.parameters()))

    # enable gradient checkpointing
    if args.enable_gc:
        gen_model.enable_gradient_checkpointing()


    # Setup optimizer and learning rate scheduler:
    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer_gen = optimizer_cls(
        gen_model_trainable_parameters,
        lr=args.learning_rate_gen,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Setup dataset:
    all_ratios = [
        '1024x1024 ( 1:1 index_0 )',
        '1152x896 ( 9:7 index_1 )',
        '896x1152 ( 7:9 index_2 )',
        '1152x864 ( 4:3 index_3 )',
        '864x1152 ( 3:4 index_4 )',
        '1248x832 ( 3:2 index_5 )',
        '832x1248 ( 2:3 index_6 )',
        '1280x720 ( 16:9 index_7 )',
        '720x1280 ( 9:16 index_8 )',
        '1344x576 ( 21:9 index_9 )',
        '576x1344 ( 9:21 index_10 )'
    ]

    prompt_keys = list(extension.student_prompt_keys(evaluation=False))
    test_prompt_keys = list(extension.student_prompt_keys(evaluation=True))
    select_ratio_index = [j for j in range(len(all_ratios))]
    select_ratio = [all_ratios[i] for i in select_ratio_index]

    dataset_root = Path(__file__).resolve().parent
    train_jsonl_path = resolve_existing_path(args.data_path_train_jsonl, dataset_root)
    test_jsonl_path = resolve_existing_path(args.data_path_test_jsonl, dataset_root)

    teacher_context_provider = extension.build_teacher_context(
        DopsdExtensionContext(
            dataset_jsonl=train_jsonl_path,
            vl_model=vl_model,
            processor=processor,
            device=accelerator.device,
            dtype=inference_dtype,
            output_dir=save_dir,
            is_main_process=accelerator.is_main_process,
            embedding_fn=get_qwen3vl_zimage_prompt_embeds,
        )
    )
    if teacher_conditioning_mode == "student_text" and teacher_context_provider is not None:
        raise ValueError(
            "student_text conditioning requires the exact student embeddings; "
            "a separate teacher context provider is not allowed"
        )

    if '1024x1024 ( 1:1 index_0 )' in select_ratio:
        test_h, test_w = 1024, 1024
    else:
        test_ratio = select_ratio[0]
        test_w = int(test_ratio.split('x')[0])
        test_h = int(test_ratio.split('x')[1].split(' ')[0])

    train_dataset = TextImageDataset(
        str(train_jsonl_path),
        target_resolutions=parse_ratios(select_ratio),
        data_root=dataset_root,
    )

    train_sampler = AspectBatchSampler(
        buckets=train_dataset.buckets,
        batch_size=args.batch_size,
        target_resolutions=parse_ratios(select_ratio),
        prompt_keys=prompt_keys,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True
    )

    num_samples = len(train_dataset)
    local_batch_size = int(args.batch_size)

    # Create data loaders:
    train_dataloader = CustomDataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # validation dataset
    num_test_samples = args.batch_size_test * accelerator.num_processes
    test_dataset = TextPromptDataset(
        str(test_jsonl_path),
        prompt_keys=test_prompt_keys,
        num_prompts=num_test_samples,
        have_gt=True,
        data_root=dataset_root,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size_test,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False
    )

    # printing
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {num_samples} samples")
        logger.info(
            f"Total batch size: {local_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
        logger.info(
            f"Total trainable parameters in gen_model: {sum(p.numel() for p in gen_model.parameters() if p.requires_grad)}")

        log_gen = os.path.join(save_dir, "loss_log", "loss_gen_log.jsonl")
        os.makedirs(os.path.dirname(log_gen), exist_ok=True)

    # prepare file to log loss
    if accelerator.is_main_process:
        # clean the log files if they exist
        if os.path.exists(log_gen):
            os.remove(log_gen)
        # add a header to the log files
        with open(log_gen, 'w') as f:
            f.write("loss for few step generator\n")

    assert get_active_deepspeed_plugin(accelerator.state) is zero2_plugin_a
    gen_model, optimizer_gen, test_dataloader = accelerator.prepare(
        gen_model, optimizer_gen, test_dataloader
    )
    extension.prepared_adapter_state(accelerator.unwrap_model(gen_model))
    d_step_observer_session = (
        None
        if d_step_observer is None
        else DStepObserverSession(
            d_step_observer,
            module=gen_model,
            optimizer=optimizer_gen,
            accelerator=accelerator,
            gradient_accumulation_steps=int(args.gradient_accumulation_steps),
            max_grad_norm=float(args.max_grad_norm),
        )
    )

    global_step = 0
    epoch_start = -1
    # resume (we now leave it blank, users can add their own checkpoints)

    if accelerator.is_main_process:
        logger.info(f"Starting training experiment: {args.exp_name}")

    if args.teacher_target_mode != "raw":
        raise ValueError("D-OPSD supports only the raw teacher target")
    if args.teacher_target_domain not in {"x0", "v"}:
        raise ValueError("teacher target domain must be x0 or v")

    teacher_timestep_indices = parse_teacher_timestep_indices(
        args.teacher_timestep_indices,
        args.num_training_steps,
    )
    teacher_timestep_warmup_indices = parse_teacher_timestep_indices(
        args.teacher_timestep_warmup_indices,
        args.num_training_steps,
    )
    training_timesteps = parse_training_timesteps(
        args.training_timesteps,
        args.num_training_steps,
    )
    adaptive_teacher_scores = {index: None for index in sorted(teacher_timestep_indices)}
    if accelerator.is_main_process:
        logger.info(
            "Teacher timestep indices for D-OPSD loss: "
            f"{','.join(str(index) for index in sorted(teacher_timestep_indices))}"
        )
        logger.info(
            "Training timestep grid: "
            f"{','.join(f'{timestep:g}' for timestep in training_timesteps)}"
        )
        logger.info(
            "Teacher target: "
            f"variant={args.teacher_target_variant} "
            f"mode={args.teacher_target_mode} "
            f"domain={args.teacher_target_domain} "
            f"gamma={args.teacher_target_gamma}"
        )
        if int(args.teacher_timestep_warmup_steps) > 0:
            logger.info(
                "Teacher timestep warmup: "
                f"steps={args.teacher_timestep_warmup_steps} "
                f"indices={','.join(str(index) for index in sorted(teacher_timestep_warmup_indices))}"
            )
        if int(args.teacher_timestep_adaptive_top_k) > 0:
            logger.info(
                "Adaptive teacher timestep selection: "
                f"top_k={args.teacher_timestep_adaptive_top_k} "
                f"warmup_steps={args.teacher_timestep_adaptive_warmup_steps} "
                f"metric={args.teacher_timestep_adaptive_metric} "
                f"ema={args.teacher_timestep_adaptive_ema}"
            )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    ############################################### Train Loop ######################################################

    if not args.disable_training_samples:
        # get sample prompts, free to change
        test_prompts, gt_image_paths = next(iter(test_dataloader))
        test_images_gt = []
        for image_path in gt_image_paths:
            with Image.open(image_path) as img:
                test_images_gt.append(img.convert("RGB"))

        with torch.no_grad():
            generator_test = create_generator(test_prompts, 2026)
            # sample multistep images for comparison
            pipeline.vae.to(accelerator.device, dtype=inference_dtype)
            with accelerator.autocast():
                with pipeline.transformer.disable_adapter() if args.use_lora > 1 else torch.no_grad():
                    images = pipeline(
                        prompt=test_prompts,
                        height=test_h,
                        width=test_w,
                        num_inference_steps=9  if args.num_training_steps < 10 else 50, # This actually results in 8 DiT forwards when set to 9
                        guidance_scale=0.0 if args.num_training_steps < 10 else 4.0,
                        generator=generator_test,
                        output_type="pt",
                    )[0]

            # resize to 1/2 resolution according to its original size
            images = torch.nn.functional.interpolate(images, size=(test_h // 2, test_w // 2), mode='bicubic',
                                                     align_corners=False)

            # Save images locally
            accelerator.wait_for_everyone()
            out_samples = accelerator.gather(images.to(torch.float32))

            pipeline.vae.to(accelerator.device, dtype=vae_dtype)

            # Save as grid images
            out_samples = Image.fromarray(array2grid(out_samples))
            if accelerator.is_main_process:
                base_dir = os.path.join(args.output_dir, args.exp_name)
                sample_dir = os.path.join(base_dir, "samples")
                os.makedirs(sample_dir, exist_ok=True)
                out_samples.save(f"{sample_dir}/samples_original.png")
                logger.info(f"Saved original sample images to {sample_dir}/samples_original.png")

    grad_norm = 0
    logs = {}
    micro_batch_index = 0
    for epoch in range(epoch_start + 1, args.epochs):
        for batch in train_dataloader:

            if global_step > 1000:
                args.sample_steps = 500

            with accelerator.accumulate(gen_model):

                gradient_accumulation_microstep = (
                    micro_batch_index % int(args.gradient_accumulation_steps)
                )


                images = batch["pixel_values"].to(device=accelerator.device, dtype=vae_dtype)
                train_dtype = inference_dtype
                prompts = batch["prompts"]
                extension.validate_student_prompts(
                    prompts,
                    batch.get("source_row_ids"),
                )


                images_vl = (images + 1) / 2
                images_vl = list(images_vl.unbind(dim=0))

                bsz = images.shape[0]
                h, w = images.shape[2], images.shape[3]


                #change to list of tensor for timesteps range from (0~1) equal /1000
                timesteps = [torch.tensor(t, device=accelerator.device, dtype=train_dtype) for t in training_timesteps]


                with torch.no_grad():
                    with accelerator.autocast():
                        prompt_embeds_list = _encode_prompt(
                            pipeline.text_encoder,
                            tokenizer,
                            prompts,
                            max_sequence_length=512,
                            device=accelerator.device,
                        )
                        prompt_embeds_list_vl = None
                        if teacher_conditioning_mode == "student_text":
                            prompt_embeds_list_vl = prompt_embeds_list
                        elif teacher_context_provider is None:
                            prompt_embeds_list_vl = get_qwen3vl_zimage_prompt_embeds(
                                vl_model=vl_model,
                                processor=processor,
                                prompts=prompts,
                                images=images_vl,
                                device=accelerator.device,
                                dtype=inference_dtype,
                                max_sequence_length=1024,
                                num_images_per_prompt=1,
                                 hidden_state_layer=-2,
                                use_system_prompt=False,
                            )

                        images = pipeline.vae.encode(images).latent_dist.mode()
                        images = (images - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor


                latents_begin = pipeline.prepare_latents(
                    batch_size=bsz,
                    num_channels_latents=num_channels_latents,
                    height=h,
                    width=w,
                    dtype=train_dtype,
                    device=accelerator.device,
                    generator=None,
                    latents=None,
                )

                latents_student = latents_begin
                latents_teacher = latents_begin

                total_loss = 0.0
                loss_dopsd_whole = []
                field_loss_records = []
                diagnostic_records = []
                student_x0_traj = []
                teacher_x0_traj = []
                optimizer_step = global_step + 1
                active_loss_indices = active_teacher_timestep_indices(
                    args,
                    optimizer_step,
                    teacher_timestep_indices,
                    teacher_timestep_warmup_indices,
                    adaptive_teacher_scores,
                )
                diagnostics_active = diagnostics.should_log_step(
                    optimizer_step,
                    accelerator.sync_gradients,
                )
                if diagnostics_active:
                    diagnostics.begin_step()

                for back_step in range(len(timesteps)):
                    t = timesteps[back_step].expand(bsz) / 1000
                    t = t.to(device=accelerator.device, dtype=train_dtype)

                    if back_step < len(timesteps) - 1:
                        next_t = timesteps[back_step + 1].expand(bsz) / 1000
                    else:
                        next_t = torch.ones_like(t)
                    next_t = next_t.to(device=accelerator.device, dtype=train_dtype)

                    dt = next_t - t

                    # detach current state to avoid cross-timestep BPTT
                    latents_student = latents_student.detach().requires_grad_(True)
                    latents_teacher = latents_teacher.detach()

                    latents_student_in = latents_student.unsqueeze(2)
                    latents_student_list = list(latents_student_in.unbind(dim=0))

                    latents_teacher_in = latents_teacher.unsqueeze(2)
                    latents_teacher_list = list(latents_teacher_in.unbind(dim=0))

                    selected_for_loss = back_step in active_loss_indices
                    teacher_forward_ms = None
                    v_pred_teacher = None
                    x_0_teacher = None
                    if selected_for_loss:
                        # teacher
                        teacher_timer = diagnostics.start_timer() if diagnostics_active else None
                        with torch.no_grad():
                            with accelerator.autocast():
                                gen_model.set_adapter("teacher")
                                teacher_prompt_embeds = prompt_embeds_list_vl
                                if teacher_conditioning_mode == "multimodal" and teacher_context_provider is not None:
                                    teacher_prompt_embeds = teacher_context_provider.context_embeddings(
                                        prompts=prompts,
                                        optimizer_step=optimizer_step,
                                        gradient_accumulation_microstep=(
                                            gradient_accumulation_microstep
                                        ),
                                        timestep_index=back_step,
                                        source_row_ids=batch.get("source_row_ids"),
                                    )
                                v_pred_teacher = gen_model(
                                    latents_student_list,
                                    t,
                                    teacher_prompt_embeds,
                                    return_dict=False,
                                )[0]
                                v_pred_teacher = torch.stack(v_pred_teacher, dim=0).squeeze(2)
                        teacher_forward_ms = diagnostics.stop_timer_ms(teacher_timer) if diagnostics_active else None

                        with torch.no_grad():
                            latents_teacher_cur = latents_student
                            x_0_teacher = latents_teacher_cur + (1 - t.reshape(bsz, 1, 1, 1)) * v_pred_teacher
                            latents_teacher = latents_teacher_cur + v_pred_teacher * dt.reshape(bsz, 1, 1, 1)

                        if d_step_observer_session is not None:
                            if args.teacher_target_domain == "x0":
                                stopped_teacher_target = x_0_teacher
                            elif args.teacher_target_domain == "v":
                                stopped_teacher_target = v_pred_teacher
                            else:
                                raise ValueError(
                                    f"Unknown teacher target domain: {args.teacher_target_domain}"
                                )
                            source_row_ids = batch.get("source_row_ids")
                            d_step_observer_session.on_stopped_microbatch(
                                step=optimizer_step,
                                microbatch=gradient_accumulation_microstep,
                                coordinate={
                                    "epoch": epoch,
                                    "micro_batch_index": micro_batch_index,
                                    "back_step": back_step,
                                    "target_domain": args.teacher_target_domain,
                                    "source_row_ids": (
                                        ()
                                        if source_row_ids is None
                                        else tuple(source_row_ids)
                                    ),
                                },
                                state=latents_student,
                                tau=t,
                                student_context=prompt_embeds_list,
                                teacher_target=stopped_teacher_target,
                            )

                    # student
                    student_timer = diagnostics.start_timer() if diagnostics_active else None
                    with accelerator.autocast():
                        gen_model.set_adapter("student")
                        v_pred_student = gen_model(
                            latents_student_list,
                            t,
                            prompt_embeds_list,
                            return_dict=False,
                        )[0]
                        v_pred_student = torch.stack(v_pred_student, dim=0).squeeze(2)
                    student_forward_ms = diagnostics.stop_timer_ms(student_timer) if diagnostics_active else None

                    latents_student_cur = latents_student
                    x_0_student = latents_student_cur + (1 - t.reshape(bsz, 1, 1, 1)) * v_pred_student
                    latents_student = latents_student_cur + v_pred_student * dt.reshape(bsz, 1, 1, 1)

                    loss_dopsd = None
                    if selected_for_loss:
                        if args.teacher_target_domain == "x0":
                            field_student = x_0_student
                            field_teacher = x_0_teacher
                        elif args.teacher_target_domain == "v":
                            field_student = v_pred_student
                            field_teacher = v_pred_teacher
                        else:
                            raise ValueError(f"Unknown teacher target domain: {args.teacher_target_domain}")
                        field_loss_records.append(
                            {
                                "back_step": back_step,
                                "field_student": field_student,
                                "field_teacher": field_teacher,
                                "x0_student": x_0_student,
                                "x0_teacher": x_0_teacher,
                                "v_pred_student": v_pred_student,
                                "v_pred_teacher": v_pred_teacher,
                            }
                        )

                    diagnostic_records.append(
                        {
                            "back_step": back_step,
                            "timestep": t.detach(),
                            "dt": dt.detach(),
                            "selected_for_loss": selected_for_loss,
                            "loss_dopsd": loss_dopsd,
                            "x0_student": x_0_student.detach(),
                            "x0_teacher": x_0_teacher.detach() if x_0_teacher is not None else None,
                            "v_pred_student": v_pred_student.detach(),
                            "v_pred_teacher": v_pred_teacher.detach() if v_pred_teacher is not None else None,
                            "teacher_forward_ms": teacher_forward_ms,
                            "student_forward_ms": student_forward_ms,
                        }
                    )

                    if (
                        not args.disable_training_samples
                        and selected_for_loss
                        and accelerator.sync_gradients
                        and ((global_step + 1) % args.sample_steps == 0)
                    ):
                        student_x0_traj.append(x_0_student.detach())
                        teacher_x0_traj.append(x_0_teacher.detach())

                if not field_loss_records:
                    raise ValueError("No teacher timesteps were selected for D-OPSD loss")

                field_targets = [
                    record["field_teacher"] for record in field_loss_records
                ]
                loss_by_back_step = {}
                for record, field_target in zip(
                    field_loss_records,
                    field_targets,
                ):
                    loss_dopsd = F.mse_loss(record["field_student"], field_target, reduction="mean")
                    total_loss = total_loss + loss_dopsd
                    loss_dopsd_whole.append(loss_dopsd.detach())
                    back_step = record["back_step"]
                    loss_by_back_step[back_step] = loss_dopsd

                    if args.teacher_timestep_adaptive_metric == "loss_x0":
                        adaptive_metric_value = loss_dopsd.detach()
                    elif args.teacher_timestep_adaptive_metric == "gap_x0_mse":
                        adaptive_metric_value = F.mse_loss(
                            record["x0_student"].detach(),
                            record["x0_teacher"].detach(),
                            reduction="mean",
                        )
                    elif args.teacher_timestep_adaptive_metric == "gap_v_mse":
                        adaptive_metric_value = F.mse_loss(
                            record["v_pred_student"].detach(),
                            record["v_pred_teacher"].detach(),
                            reduction="mean",
                        )
                    else:
                        adaptive_metric_value = None
                    update_adaptive_teacher_score(
                        args,
                        adaptive_teacher_scores,
                        back_step,
                        adaptive_metric_value,
                    )

                if diagnostics_active:
                    for record in diagnostic_records:
                        back_step = record["back_step"]
                        loss_dopsd = loss_by_back_step.get(back_step)
                        diagnostics.record_timestep(
                            optimizer_step=optimizer_step,
                            epoch=epoch,
                            back_step=back_step,
                            timestep=record["timestep"],
                            dt=record["dt"],
                            selected_for_loss=record["selected_for_loss"],
                            loss_x0=loss_dopsd.detach() if loss_dopsd is not None else None,
                            x0_student=record["x0_student"],
                            x0_teacher=record["x0_teacher"],
                            v_pred_student=record["v_pred_student"],
                            v_pred_teacher=record["v_pred_teacher"],
                            teacher_forward_ms=record["teacher_forward_ms"],
                            student_forward_ms=record["student_forward_ms"],
                            teacher_target_variant=args.teacher_target_variant,
                            teacher_target_mode=args.teacher_target_mode,
                            teacher_target_domain=args.teacher_target_domain,
                            teacher_target_gamma=args.teacher_target_gamma,
                        )

                total_loss = total_loss / len(loss_dopsd_whole)
                if d_step_observer_session is not None:
                    d_step_observer_session.record_microbatch_loss(
                        step=optimizer_step,
                        microbatch=gradient_accumulation_microstep,
                        loss=total_loss,
                    )
                extension_auxiliary_loss = extension.auxiliary_loss(
                    gen_model=gen_model,
                    pipeline=pipeline,
                    accelerator=accelerator,
                    optimizer_step=optimizer_step,
                    micro_step=gradient_accumulation_microstep,
                )
                if extension_auxiliary_loss is not None:
                    total_loss = total_loss + extension_auxiliary_loss
                backward_timer = diagnostics.start_timer() if diagnostics_active else None
                accelerator.backward(total_loss)
                backward_ms = diagnostics.stop_timer_ms(backward_timer) if diagnostics_active else None

                grad_norm = None
                if accelerator.sync_gradients:
                    if (
                        d_step_observer_session is not None
                        and not d_step_observer_session.owns_optimizer_step_hooks
                    ):
                        unscale_result = accelerator.unscale_gradients(optimizer_gen)
                        if unscale_result is not None:
                            raise TypeError(
                                "canonical D unscale operation must return None"
                            )
                        d_step_observer_session.on_pre_clip(step=optimizer_step)
                    grad_norm = accelerator.clip_grad_norm_(gen_model.parameters(), args.max_grad_norm)
                    if (
                        d_step_observer_session is not None
                        and not d_step_observer_session.owns_optimizer_step_hooks
                    ):
                        d_step_observer_session.on_post_clip(
                            step=optimizer_step,
                            clip_branch=observed_clip_branch(
                                grad_norm, args.max_grad_norm
                            ),
                        )

                optimizer_gen.step()
                if (
                    d_step_observer_session is not None
                    and accelerator.sync_gradients
                    and not d_step_observer_session.owns_optimizer_step_hooks
                ):
                    d_step_observer_session.on_post_optimizer(step=optimizer_step)
                optimizer_gen.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    global_step += 1
                    progress_bar.update(1)

                    logs = {
                        "loss_dopsd": accelerator.gather(torch.stack(loss_dopsd_whole).detach()).mean().item(),
                        "loss_total": accelerator.gather(total_loss.detach()).mean().item(),
                        "glo_s": global_step,
                        "epoch": epoch,
                        "grad_n": float(grad_norm) if grad_norm is not None else 0.0,
                    }

                    accelerator.log(logs, step=global_step)
                    if diagnostics_active:
                        diagnostics.finalize_step(
                            optimizer_step=global_step,
                            backward_ms=backward_ms,
                            total_loss=total_loss.detach(),
                            grad_norm=float(grad_norm) if grad_norm is not None else 0.0,
                        )
                    if teacher_adapter_update_mode == "ema":
                        ema_update_lora_adapter(
                            gen_model,
                            src_adapter="student",
                            dst_adapter="teacher",
                            ema_decay=args.ema_decay,
                        )

                    if d_step_observer_session is not None:
                        d_step_observer_session.on_post_ema(step=global_step)

                    extension.on_optimizer_step_end(
                        gen_model=gen_model,
                        pipeline=pipeline,
                        accelerator=accelerator,
                        optimizer_step=global_step,
                    )

                    if accelerator.is_main_process:
                        with open(log_gen, "a") as f_log_gen:
                            f_log_gen.write(f"{json.dumps(logs)}\n")

                    # save model
                    if global_step % args.checkpoint_steps == 0 or global_step == args.max_train_steps:
                        extension.validate_teacher_adapter_state(
                            unwrap_model(gen_model, accelerator),
                            optimizer_step=global_step,
                            event="checkpoint",
                        )
                        # save checkpoint
                        if accelerator.is_main_process:
                            if args.use_lora > 1:
                                lora_dict_gen = os.path.join(checkpoint_dir, f"lora_gen_step_{global_step}")
                                os.makedirs(lora_dict_gen, exist_ok=True)
                                unwrap_model(gen_model, accelerator).save_pretrained(lora_dict_gen)
                            else:
                                ckpt_dict_gen = os.path.join(checkpoint_dir, f"gen_model_step_{global_step}.pt")
                                accelerator.save(unwrap_model(gen_model, accelerator).state_dict(), ckpt_dict_gen)
                                logger.info(f"Saved model checkpoint to {checkpoint_dir} at step {global_step}")

                    # visualize samples
                    if (
                        not args.disable_training_samples
                        and (global_step % args.sample_steps == 0 or global_step == args.max_train_steps)
                    ):
                        with torch.no_grad():
                            pipeline.vae.to(accelerator.device, dtype=inference_dtype)

                            traj_dir = os.path.join(args.output_dir, args.exp_name)
                            traj_dir = os.path.join(traj_dir, "samples_trajectory")
                            save_student_teacher_trajectory(
                                pipeline,
                                student_x0_traj,
                                teacher_x0_traj,
                                traj_dir,
                                global_step,
                                accelerator,
                                max_size=(2048, 2048),
                            )

                            # sample multistep images for comparison
                            gen_model.set_adapter("student")
                            with accelerator.autocast():
                                images_s = pipeline(
                                    prompt=test_prompts,
                                    height=test_h,
                                    width=test_w,
                                    num_inference_steps=9 if args.num_training_steps < 10 else 50,
                                    # This actually results in 8 DiT forwards when set to 9
                                    guidance_scale=0.0 if args.num_training_steps < 10 else 4.0,
                                    generator=generator_test,
                                    output_type="pt",
                                )[0]

                            images_s = torch.nn.functional.interpolate(images_s, size=(test_h // 2, test_w // 2),
                                                                     mode='bicubic', align_corners=False)

                            gen_model.set_adapter("teacher")
                            with accelerator.autocast():
                                prompt_embeds_vl_test = get_qwen3vl_zimage_prompt_embeds(
                                    vl_model=vl_model,
                                    processor=processor,
                                    prompts=test_prompts,
                                    images=test_images_gt,
                                    device=accelerator.device,
                                    dtype=inference_dtype,
                                    max_sequence_length=1024,
                                    num_images_per_prompt=1,
                                     hidden_state_layer=-2,
                                    use_system_prompt=False,
                        )
                                images_t = pipeline(
                                    prompt_embeds=prompt_embeds_vl_test,
                                    height=test_h,
                                    width=test_w,
                                    num_inference_steps=9 if args.num_training_steps < 10 else 50,
                                    # This actually results in 8 DiT forwards when set to 9
                                    guidance_scale=0.0 if args.num_training_steps < 10 else 4.0,
                                    # Guidance should be 0 for the Turbo models
                                    generator=generator_test,
                                    output_type="pt",
                                )[0]
                            image_t = torch.nn.functional.interpolate(images_t, size=(test_h // 2, test_w // 2), mode='bicubic', align_corners=False)

                            # Save images locally
                            accelerator.wait_for_everyone()
                            out_samples = accelerator.gather(images_s.to(torch.float32))
                            out_samples_t = accelerator.gather(image_t.to(torch.float32))

                            # Save as grid images
                            out_samples = Image.fromarray(array2grid(out_samples))
                            out_samples_t = Image.fromarray(array2grid(out_samples_t))
                            if accelerator.is_main_process:

                                base_dir = os.path.join(args.output_dir, args.exp_name)
                                sample_dir = os.path.join(base_dir, "samples")
                                os.makedirs(sample_dir, exist_ok=True)
                                out_samples.save(f"{sample_dir}/samples_step_{global_step}_student.png")
                                out_samples_t.save(f"{sample_dir}/samples_step_{global_step}_teacher.png")
                                logger.info(f"Saved sample images to {sample_dir}/samples_step_{global_step}.png")

                            pipeline.vae.to(accelerator.device, dtype=vae_dtype)
            if logs:
                progress_bar.set_postfix(**logs)
            micro_batch_index += 1

            ############################################### End Train Loop ######################################################

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Training completed.")
    if teacher_context_provider is not None:
        teacher_context_provider.close()
    diagnostics.close()
    accelerator.end_training()


def main(args):
    run_training(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)



































