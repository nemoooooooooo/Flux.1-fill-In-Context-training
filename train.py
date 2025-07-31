from args_parser import parse_args
import torch
import shutil
from pathlib import Path 
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from accelerate import Accelerator
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
import logging
from accelerate.logging import get_logger
import transformers
import diffusers
import os
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from utils import import_model_class_from_model_name_or_path, CustomFlowMatchEulerDiscreteScheduler, load_text_encoders
import copy
from diffusers import (
    AutoencoderKL,
    FluxFillPipeline,
    FluxTransformer2DModel,
)
from peft import LoraConfig, set_peft_model_state_dict
from utils import PairedImageDataset, collate_fn, encode_prompt, prepare_mask_latents
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils.torch_utils import is_compiled_module
from typing import List, Union
import math
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from peft.utils import get_peft_model_state_dict
from contextlib import nullcontext
from torchvision import transforms
from huggingface_hub import create_repo, upload_folder

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.32.0.dev0")

logger = get_logger(__name__)

def log_validation(pipeline, args, accelerator, dataloader, tag="validation"):
    logger.info(f"Running {tag}...")
    pipeline = pipeline.to(accelerator.device)

    if accelerator.mixed_precision == "bf16":
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    elif accelerator.mixed_precision == "fp16":
        autocast_ctx = torch.autocast("cuda", dtype=torch.float16)
    else:
        autocast_ctx = nullcontext()

    gen = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    logged = []
    with autocast_ctx:
        for batch in dataloader:
            prompt = batch["prompts"]
            image  = batch["pixel_values"]            # source (inpaint) image if you need it
            mask   = batch["mask_pixel_values"]       # mask tensor

            # Convert mask/image tensors back to PIL for pipeline if needed
            # (FluxFillPipeline usually takes PIL/np arrays)
            pil_img  = [transforms.ToPILImage()(img.cpu()*0.5+0.5) for img in image]
            pil_mask = [transforms.ToPILImage()(m.squeeze(0).cpu()) for m in mask]

            out = pipeline(
                prompt=prompt,
                image=pil_img,
                mask_image=pil_mask,
                height=1024,
                width=768,
                num_inference_steps=28,
                guidance_scale=30,
                generator=gen,
            ).images

            logged.append((pil_img[0], pil_mask[0], out[0], prompt[0]))

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            wandb_imgs = []
            for src, m, gen, pr in logged:
                wandb_imgs.extend(
                    [
                        wandb.Image(src, caption="source"),
                        wandb.Image(m,   caption="mask"),
                        wandb.Image(gen, caption=pr),
                    ]
                )
            tracker.log({tag: wandb_imgs})

    # cleanup
    del pipeline, logged
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )


    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        model_id = args.hub_model_id or Path(args.output_dir).name
        repo_id = None
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=model_id,
                exist_ok=True,
            ).repo_id

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = CustomFlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two = load_text_encoders(args, text_encoder_cls_one, text_encoder_cls_two)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )

    model_path = args.base_transformer if args.base_transformer is not None else args.pretrained_model_name_or_path
    transformer = FluxTransformer2DModel.from_pretrained(
        model_path, 
        subfolder="transformer", 
        revision=args.revision, 
        variant=args.variant
    )

    # Freeze the models
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # -------------------------------------------------------------------
    #  VAE constants that other helpers need
    # -------------------------------------------------------------------
    vae_shift   = vae.config.shift_factor
    vae_scale   = vae.config.scaling_factor
    vae_channels = vae.config.block_out_channels      # later for unpack/pack



    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()


    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]


    # Add new LoRA weights to the attention layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )

    transformer.add_adapter(transformer_lora_config)
    transformer.to(accelerator.device, dtype=weight_dtype)   # move new adapters

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            FluxFillPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = FluxFillPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )


        # Make sure the trainable params are in float32. This is again needed since the base models are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    freeze_text_encoder = True
    pure_textual_inversion = False

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]


    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                print("using 8bit adam optimizer")
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Dataset and PairedImageDataset creation:
    train_dataset = PairedImageDataset(
        args=args,
        split="train",
    )
    val_dataset = PairedImageDataset(
        args=args,
        split=args.val_split,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,          # no lambda, no extra arg
        num_workers=1,
    )
    validation_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn, 
        num_workers=1,
    )

    if freeze_text_encoder:
        # --------------------------------------------------------------------------
        # Prompt handling
        #   • We never finetune a text-encoder, so embeddings can be cached
        #     when every sample uses the same prompt.
        #   • If the dataset provides per-image captions (`use_caption=True`)
        #     we will recompute embeddings inside the training loop instead.
        # --------------------------------------------------------------------------
        def compute_prompt_embeddings(prompt: Union[str, List[str]]):
            """
            Wrapper around `encode_prompt` that
            1. calls both encoders,
            2. moves everything to the accelerator device, and
            3. returns (prompt_embeds, pooled_embeds, text_ids)
            """
            with torch.no_grad():
                pe, pooled, tids = encode_prompt(
                    text_encoders=[text_encoder_one, text_encoder_two],
                    tokenizers=[tokenizer_one, tokenizer_two],
                    prompt=prompt,
                    max_sequence_length=args.max_sequence_length,
                )
                return (
                    pe.to(accelerator.device),
                    pooled.to(accelerator.device),
                    tids.to(accelerator.device),
                )
            
        # --------------------------------------------------------------------------
        # STATIC cache (single instance prompt, no captions)
        # --------------------------------------------------------------------------
        prompt_cache = None
        if freeze_text_encoder and not train_dataset.use_caption:
            inst_pe, inst_pool, inst_ids = compute_prompt_embeddings(args.instance_prompt)
            prompt_cache = {"prompt": inst_pe, "pooled": inst_pool, "ids": inst_ids}

            # free memory held by heavy encoders once we have the cache
            del tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two
            free_memory()


        def get_batch_prompt_embeddings(batch_prompts, batch_size):
            """
            Returns embeddings for the current minibatch.
            • When `prompt_cache` exists we just repeat the cached tensors.
            • Otherwise we recompute on-the-fly (caption case).
            """
            if prompt_cache is not None:
                # cache holds embeddings for **one** prompt; repeat to match batch
                return (
                    prompt_cache["prompt"].repeat(batch_size, 1, 1),
                    prompt_cache["pooled"].repeat(batch_size, 1),
                    prompt_cache["ids"].repeat(batch_size, 1, 1),
                )
            else:
                return compute_prompt_embeddings(batch_prompts)



    # -------------------------------------------------------------------
    #  How many optimisation steps & LR schedule
    # -------------------------------------------------------------------
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )

    overrode_max_train_steps = False
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,                       # e.g. "constant_with_warmup"
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # -----------------------------------------------------------
    # Send the objects to the Accelerator (only the pieces that are trainable / iterate in the loop)
    # -----------------------------------------------------------
    transformer, optimizer, train_dataloader, validation_dataloader, lr_scheduler = (
        accelerator.prepare(
            transformer,
            optimizer,
            train_dataloader,
            validation_dataloader,   # ← include val loader so it’s on the right device/process group
            lr_scheduler,
        )
    )

    # -----------------------------------------------------------
    # → recalc how many optimisation steps we really have
    # -----------------------------------------------------------
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )

    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Re-derive number of *whole* epochs we will actually run
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch
    )

    # -----------------------------------------------------------
    # initialise experiment trackers
    # -----------------------------------------------------------
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="lora-only-flux",
            config=vars(args),        # logs full CLI config
        )

    # ------------------------------------------------------------------
    # Training bookkeeping & (optional) checkpoint resume 
    # ------------------------------------------------------------------
    total_batch_size = (
        args.batch_size                     # per-device batch
        * accelerator.num_processes         # world size
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step  = 0
    first_epoch  = 0
    initial_global_step = 0

    # ---------------------------------------------------------------
    #  Resume from checkpoint (optional)
    # ---------------------------------------------------------------
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            ckpt_dir = os.path.basename(args.resume_from_checkpoint)
        else:
            all_ckpts = sorted(
                (d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")),
                key=lambda x: int(x.split("-")[1]),
            )
            ckpt_dir = all_ckpts[-1] if all_ckpts else None

        if ckpt_dir is None:
            logger.info(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist – starting fresh."
            )
            args.resume_from_checkpoint = None
        else:
            logger.info(f"Resuming from checkpoint {ckpt_dir}")
            accelerator.load_state(os.path.join(args.output_dir, ckpt_dir))
            global_step  = int(ckpt_dir.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    # ---------------------------------------------------------------
    #  Progress bar
    # ---------------------------------------------------------------
    progress_bar = tqdm(
        range(initial_global_step, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # ---------------------------------------------------------------
    #  Helper to fetch σₜ values for the custom scheduler
    # ---------------------------------------------------------------
    def get_sigmas(timesteps, n_dim: int = 4, dtype: torch.dtype = torch.float32):
        """
        Map a batch of 'flow-matching' timesteps (0-1000) to their σ
        as done in the original script.
        """
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_ts = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps   = timesteps.to(accelerator.device)

        idx = [(schedule_ts == t).nonzero().item() for t in timesteps]
        sigma = sigmas[idx].flatten()
        while sigma.ndim < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # -----------------------------------------------
    # inside the main training loop
    # -----------------------------------------------
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()                       # LoRA layers are the only trainables

        for step, batch in enumerate(train_dataloader):

            with accelerator.accumulate(transformer):
                # ──────────────────────────────
                # 1. prompts  → embeddings
                # ──────────────────────────────
                prompts       = batch["prompts"]              # list[str]  len = current mini-batch
                bsz           = len(prompts)
                prompt_embeds, pooled_embeds, text_ids = get_batch_prompt_embeddings(
                    prompts, batch_size=bsz
                )                                             # (B,S,*) / (B,*) / (B,S,3)
                # ──────────────────────────────
                # 2. pixels   → latent z₀
                # ──────────────────────────────
                pixel_values = batch["pixel_values"].to(         
                    dtype = vae.dtype,
                    device = accelerator.device,
                )


                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae_shift) * vae_scale
                model_input = model_input.to(dtype = weight_dtype)

                # ──────────────────────────────
                # 3. mask   → mask latents
                # ──────────────────────────────
                mask         = batch["mask_pixel_values"].to(
                    accelerator.device, dtype=vae.dtype
                )                                                         # 1×H×W
                masked_image = pixel_values * (1.0 - mask)

                mask_lat, masked_img_lat = prepare_mask_latents(
                    vae                = vae,
                    mask               = mask,
                    masked_image       = masked_image,
                    batch_size         = bsz,
                    num_channels_latents = 16,
                    num_images_per_prompt = 1,
                    height             = pixel_values.shape[2],
                    width              = pixel_values.shape[3],
                    dtype              = model_input.dtype,
                    device             = accelerator.device,
                )
                masked_img_lat = torch.cat([masked_img_lat, mask_lat], dim=-1)

                # ──────────────────────────────
                # 4. flow-matching timestep & noise
                # ──────────────────────────────
                noise    = torch.randn_like(model_input)
                u        = compute_density_for_timestep_sampling(
                    weighting_scheme = args.weighting_scheme,
                    batch_size       = bsz,
                    logit_mean       = args.logit_mean,
                    logit_std        = args.logit_std,
                    mode_scale       = args.mode_scale,
                )
                indices  = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(model_input.device)

                sigma    = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_in = (1.0 - sigma) * model_input + sigma * noise      # z_t

                # ──────────────────────────────
                # 5. pack latents & ids
                # ──────────────────────────────
                packed_noisy = FluxFillPipeline._pack_latents(
                    noisy_in,
                    batch_size         = bsz,
                    num_channels_latents = noisy_in.shape[1],
                    height             = noisy_in.shape[2],
                    width              = noisy_in.shape[3],
                )

                latent_ids = FluxFillPipeline._prepare_latent_image_ids(
                    bsz,
                    noisy_in.shape[2] // 2,
                    noisy_in.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )

                # ──────────────────────────────
                # 6. forward
                # ──────────────────────────────
                if transformer.config.guidance_embeds:
                    guidance = torch.full((bsz,), args.guidance_scale,
                                        device=accelerator.device)
                else:
                    guidance = None

                pred = transformer(
                    hidden_states       = torch.cat([packed_noisy, masked_img_lat], dim=2),
                    timestep            = timesteps / 1000.0,      # scale like original
                    guidance            = guidance,
                    pooled_projections  = pooled_embeds,
                    encoder_hidden_states = prompt_embeds,
                    txt_ids             = text_ids,
                    img_ids             = latent_ids,
                    return_dict         = False,
                )[0]

                pred = FluxFillPipeline._unpack_latents(
                    pred,
                    height            = noisy_in.shape[2] * (2 ** (len(vae_channels)-1)),
                    width             = noisy_in.shape[3] * (2 ** (len(vae_channels)-1)),
                    vae_scale_factor  = 2 ** (len(vae_channels)-1),
                )

                # ──────────────────────────────
                # 7. loss & backward
                # ──────────────────────────────
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme = args.weighting_scheme,
                    sigmas           = sigma,
                )
                target    = noise - model_input
                loss      = ((weighting * (pred - target) ** 2)
                            .reshape(bsz, -1)
                            .mean())

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    # gradient-clipping (same threshold as original)
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)

                optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()

            # ──────────────────────────────
            # 8. book-keeping
            # ──────────────────────────────
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # ─── Periodic checkpoints ───────────────────────────────────────
                if accelerator.is_main_process and (global_step % args.checkpointing_steps == 0):
                    # ‣ apply retention limit
                    if args.checkpoints_total_limit is not None:
                        ckpts = sorted(
                            (d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")),
                            key=lambda x: int(x.split("-")[1]),
                        )
                        excess = len(ckpts) - args.checkpoints_total_limit + 1
                        for old in ckpts[:max(0, excess)]:
                            shutil.rmtree(os.path.join(args.output_dir, old))
                            logger.info(f"Removed old checkpoint {old}")

                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(ckpt_dir)          # triggers save_model_hook
                    logger.info(f"Saved checkpoint to {ckpt_dir}")

                # ─── Scalar logs (loss, LR) ─────────────────────────────────────
                logs = {
                    "loss": loss.detach().item(),
                    "lr":   lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                # ─── On-the-fly validation images ───────────────────────────────
                if (
                    validation_dataloader                                 # comes from accelerator.prepare(...)
                    and (global_step % args.validation_steps == 0)
                    and accelerator.is_main_process
                ):
                    # rebuild a lightweight pipeline *only* for sampling
                    pipe = FluxFillPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        transformer = accelerator.unwrap_model(transformer),   # LoRA-augmented
                        vae         = vae,
                        torch_dtype = weight_dtype,
                    )
                    pipe.load_lora_weights(args.output_dir)        # use current adapters
                    log_validation(pipe, args, accelerator, validation_dataloader, tag="validation")
                    del pipe
                    free_memory()

                # ─── Hard stop ──────────────────────────────────────────────────
                if global_step >= args.max_train_steps:
                    break

    # ────────────────────────────────────────────────────────────────────
    #  FINAL SAVE 
    # ────────────────────────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # 1. pull LoRA weights out of the wrapped model
        lora_state = get_peft_model_state_dict(unwrap_model(transformer).to(weight_dtype))

        # 2. write them to disk
        FluxFillPipeline.save_lora_weights(
            save_directory        = args.output_dir,
            transformer_lora_layers = lora_state,
        )
        logger.info(f"LoRA adapters saved to {args.output_dir}")

        # 3. optional final validation pass
        if validation_dataloader:
            pipe = FluxFillPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype = weight_dtype,
            )
            pipe.load_lora_weights(args.output_dir)
            log_validation(pipe, args, accelerator, validation_dataloader, tag="test", is_final_validation=True)
            del pipe; free_memory()

        # 4. push to Hub if requested
        if args.push_to_hub:
            upload_folder(
                repo_id        = repo_id,
                folder_path    = args.output_dir,
                commit_message = "End of training",
                ignore_patterns = ["step_*", "epoch_*"],
            )

    accelerator.end_training()



if __name__ == "__main__":
    args = parse_args()
    main(args)