from args_parser import parse_args

import copy, logging, math, os, shutil
from pathlib import Path
from contextlib import nullcontext
from typing import List, Union
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    FluxFillPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

from data_loader import PairedImageDataset, collate_fn
from utils import (
    CustomFlowMatchEulerDiscreteScheduler,
    encode_prompt,
    import_model_class_from_model_name_or_path,
    load_text_encoders,
    prepare_mask_latents,
    enable_trainables_by_name,
    write_mode_marker,
    read_mode_marker
)

if is_wandb_available():
    import wandb  # noqa: F401

check_min_version("0.32.0.dev0")
logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Validation helper
# ─────────────────────────────────────────────────────────────────────────────
def log_validation(pipeline, args, accelerator, dataloader, tag="validation", save_dir=None):
    """
    Run one pass over `dataloader`, create images, and push them to any active
    accelerator trackers (e.g. WandB).  Image *height* and *width* are taken
    from each sample itself, so generations always match the size seen during
    training buckets.
    """
    logger.info(f"Running {tag}…")
    pipeline = pipeline.to(accelerator.device)

    if accelerator.mixed_precision == "bf16":
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    elif accelerator.mixed_precision == "fp16":
        autocast_ctx = torch.autocast("cuda", dtype=torch.float16)
    else:
        autocast_ctx = nullcontext()

    # optional disk output
    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None and accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    # keep generator deterministic if a seed is provided
    gen = (
        torch.Generator(device=accelerator.device).manual_seed(args.seed)
        if args.seed is not None
        else None
    )

    logged = []
    samples_processed = 0
    with autocast_ctx:
        for batch in dataloader:
            # Check if we've reached the max samples limit
            if args.max_validation_samples and samples_processed >= args.max_validation_samples:
                logger.info(f"Reached max validation samples limit: {args.max_validation_samples}")
                break

            prompt = batch["prompts"]                      # list[str] (len == batch)
            image  = batch["pixel_values"]                 # (B, C, H, W) tensor
            mask   = batch["mask_pixel_values"]            # (B, 1, H, W) tensor

            # ──> tensor → PIL  (re-normalise from [-1,1] to [0,1])
            pil_img = [transforms.ToPILImage()(img.cpu() * 0.5 + 0.5) for img in image]
            pil_msk = [transforms.ToPILImage()(m.squeeze(0).cpu())    for m in mask]

            # ──> use bucket resolution of this sample
            h, w = pil_img[0].height, pil_img[0].width

            outs = pipeline(
                prompt       = prompt,
                image        = pil_img,
                mask_image   = pil_msk,
                height       = h,           
                width        = w,
                num_inference_steps = 30,
                guidance_scale      = 8,
                generator           = gen,
            ).images

            logged.append((pil_img[0], pil_msk[0], outs[0], prompt[0]))

            # save ALL images of the batch to disk (main process only)
            if save_dir is not None and accelerator.is_main_process:
                for i, out_img in enumerate(outs):
                    idx = samples_processed + i
                    base = save_dir / f"{idx:05d}"
                    try:
                        pil_img[i].save(base.with_name(f"val_{base.stem}_source.png"))
                        pil_msk[i].save(base.with_name(f"val_{base.stem}_mask.png"))
                        out_img.save(base.with_name(f"val_{base.stem}_result.png"))
                    except Exception as e:
                        logger.warning(f"Failed to save sample {idx}: {e}")

            samples_processed += len(prompt) 

    # ──> log to tracker
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            import wandb  # local import keeps hard dep optional
            imgs: list["wandb.Image"] = []
            for src, msk, gen, pr in logged:
                imgs.extend(
                    [
                        wandb.Image(src, caption="source"),
                        wandb.Image(msk, caption="mask"),
                        wandb.Image(gen, caption=pr),
                    ]
                )
            tracker.log({tag: imgs})

    # housekeeping
    del pipeline, logged
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Main ------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main(args):
    # ---------------------------------------------------------------------
    # Basic safety checks --------------------------------------------------
    # ---------------------------------------------------------------------
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "Cannot use report_to=wandb together with hub_token; please hg login instead."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        raise ValueError("bfloat16 mixed precision not supported on MPS")

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True    

    # ---------------------------------------------------------------------
    # Accelerator ----------------------------------------------------------
    # ---------------------------------------------------------------------
    logging_dir = Path(args.output_dir, args.logging_dir)
    acc_cfg = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=acc_cfg,
        kwargs_handlers=[ddp_kwargs],
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False  # bf16 isn't supported on MPS anyway

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    # ---------------------------------------------------------------
    # Optional TF32 enable (only if a CUDA device exists)  ← Fix #9
    # ---------------------------------------------------------------
    if args.allow_tf32 and torch.cuda.is_available():
        # Ampere / Hopper etc. can use TF32 for faster matmuls
        torch.backends.cuda.matmul.allow_tf32 = True

    # ---------------------------------------------------------------------
    # Output / Hub repo ----------------------------------------------------
    # ---------------------------------------------------------------------
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            from huggingface_hub import create_repo

            repo_id = create_repo(repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True).repo_id
    else:
        repo_id = None

    # ---------------------------------------------------------------------
    # Tokenisers & text encoders ------------------------------------------
    # ---------------------------------------------------------------------
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision
    )

    TextEnc1 = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    TextEnc2 = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    noise_sched = CustomFlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_sched_copy = copy.deepcopy(noise_sched)

    text_enc1, text_enc2 = load_text_encoders(args, TextEnc1, TextEnc2)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )

    model_path = args.base_transformer or args.pretrained_model_name_or_path
    transformer = FluxTransformer2DModel.from_pretrained(
        model_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )

    # ---------------------------------------------------------------------
    # Freeze everything but the LoRA layers --------------------------------
    # ---------------------------------------------------------------------
    for m in (vae, transformer, text_enc1, text_enc2):
        m.requires_grad_(False)

    precision_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    weight_dtype = precision_map.get(accelerator.mixed_precision, torch.float32)

    for m in (vae, transformer, text_enc1, text_enc2):
        m.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # ------------------------------------------------------------------
    # Select trainables by mode
    # ------------------------------------------------------------------
    if args.train_mode == "lora":
        targets = [x.strip() for x in args.lora_layers.split(",")] if args.lora_layers else [
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
            "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2",
        ]
        transformer.add_adapter(
            LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian", target_modules=targets)
        )

        # LoRA injects trainable params for you
        trainable_params = [p for p in transformer.parameters() if p.requires_grad]
        if accelerator.is_main_process:
            print(f"[LoRA] trainable tensors: {len(trainable_params)} "
                f"({sum(p.numel() for p in trainable_params)/1e6:.3f} M params)")

    else:
        # Base FT: enable grads only on names that start with prefixes and contain includes
        enabled_names, total_numel = enable_trainables_by_name(
            transformer,
            prefixes_str=args.trainable_name_prefixes,   # e.g. "transformer_blocks.,single_transformer_blocks."
            includes_str=args.train_includes,            # e.g. "attn"
            logger=logger if 'logger' in globals() else None,
            is_main_process=accelerator.is_main_process if 'accelerator' in globals() else True,
        )
        trainable_params = [p for p in transformer.parameters() if p.requires_grad]

        # hard fail if nothing matched (prevents “silent no-training” sessions)
        if len(trainable_params) == 0:
            raise RuntimeError(
                f"No parameters matched prefixes={args.trainable_name_prefixes} "
                f"and includes={args.train_includes}"
            )

        if accelerator.is_main_process:
            print(f"[BASE] trainable tensors: {len(trainable_params)} "
                f"({sum(p.numel() for p in trainable_params)/1e6:.3f} M params)")
            # Also print the names that were enabled
            for n in enabled_names:
                print("  ✓", n)


    def unwrap(model):
        m = accelerator.unwrap_model(model)
        return m._orig_mod if is_compiled_module(m) else m

    # ------------------------------------------------------------------
    # Save/load hooks so Accelerator checkpoints only the LoRA ---------
    # ------------------------------------------------------------------
    def save_hook(models, _, out_dir):
        if not accelerator.is_main_process:
            return
        for m in models:
            if isinstance(m, type(unwrap(transformer))):
                if args.train_mode == "lora":
                    lora_state = get_peft_model_state_dict(m)
                    FluxFillPipeline.save_lora_weights(out_dir, transformer_lora_layers=lora_state)
                else:
                    # save full transformer weights only
                    unwrap(transformer).save_pretrained(os.path.join(out_dir, "transformer"), max_shard_size="50GB")
            _ = _.pop()
        write_mode_marker(out_dir, args.train_mode)


    def load_hook(models, in_dir):
        saved_mode = read_mode_marker(in_dir)
        if saved_mode is not None and saved_mode != args.train_mode:
            raise RuntimeError(
                f"Mismatched training mode for load_hook: checkpoint directory indicates '{saved_mode}', "
                f"but current run is '{args.train_mode}'."
            )

        while models:
            mdl = models.pop()
            if isinstance(mdl, type(unwrap(transformer))):
                if args.train_mode == "lora":
                    state = FluxFillPipeline.lora_state_dict(in_dir)
                    state = {k.replace("transformer.", ""): v for k, v in state.items() if k.startswith("transformer.")}
                    state = convert_unet_state_dict_to_peft(state)
                    set_peft_model_state_dict(mdl, state, adapter_name="default")
                else:
                    loaded = FluxTransformer2DModel.from_pretrained(in_dir, subfolder="transformer")
                    mdl.register_to_config(**loaded.config)
                    mdl.load_state_dict(loaded.state_dict(), strict=True)
                    del loaded

                if args.mixed_precision == "fp16":
                    cast_training_params([mdl])

            else:
                raise ValueError("unexpected model type during load")



    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)

    # ------------------------------------------------------------------
    # Optimiser & LR sched --------------------------------------------
    # ------------------------------------------------------------------

    if args.scale_lr:
        args.learning_rate *= (
            args.gradient_accumulation_steps
            * args.batch_size
            * accelerator.num_processes
        )

    optim_groups = [{"params": trainable_params, "lr": args.learning_rate}]

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            import bitsandbytes as bnb
            opt_cls = bnb.optim.AdamW8bit
        else:
            opt_cls = torch.optim.AdamW
        optimizer = opt_cls(
            optim_groups,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:  # prodigy
        import prodigyopt

        optimizer = prodigyopt.Prodigy(
            optim_groups,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )


    # Dataset -----------------------------------------------------------
    train_ds = PairedImageDataset(args, split="train")
    val_ds = PairedImageDataset(args, split=args.val_split)

    dl_train = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        collate_fn=collate_fn,
    )
    dl_val = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )


    # ---- prompt caching ----------------------------------------------
    def compute_prompt_emb(prompt: Union[str, List[str]]):
        with torch.no_grad():
            pe, pool, ids = encode_prompt(
                [text_enc1, text_enc2], [tokenizer_one, tokenizer_two], prompt, args.max_sequence_length
            )
            return pe.to(accelerator.device), pool.to(accelerator.device), ids.to(accelerator.device)

    prompt_cache = None
    if not train_ds.use_caption:
        prompt_cache = {
            "prompt": compute_prompt_emb(args.instance_prompt)[0],
            "pooled": compute_prompt_emb(args.instance_prompt)[1],
            "ids": compute_prompt_emb(args.instance_prompt)[2],
        }
        # only free heavy encoders if we *never* need captions
        if prompt_cache is not None:
            del tokenizer_one, tokenizer_two, text_enc1, text_enc2  # type: ignore
            free_memory()

    def batch_prompt_emb(prompts, b):
        if prompt_cache is not None:
            return (
                prompt_cache["prompt"].repeat(b, 1, 1),
                prompt_cache["pooled"].repeat(b, 1),
                prompt_cache["ids"].repeat(b, 1, 1),
            )
        return compute_prompt_emb(prompts)

    # ---- Accelerator prepare -----------------------------------------
    transformer, optimizer, dl_train, dl_val = accelerator.prepare(
        transformer, optimizer, dl_train, dl_val
    )

    # ---- sched & epoch math ------------------------------------------
    steps_per_epoch = math.ceil(len(dl_train) / args.gradient_accumulation_steps)
    max_steps = args.max_train_steps or args.num_train_epochs * steps_per_epoch
    if args.num_train_epochs is None or args.num_train_epochs <= 0:
        args.num_train_epochs = math.ceil(max_steps / steps_per_epoch)

    lr_sched = get_scheduler(
        args.lr_scheduler,
        optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # ---- tracking & checkpoints --------------------------------------
    if accelerator.is_main_process:
        accelerator.init_trackers(
            args.tracker_project_name,
            config=vars(args))

    global_step, first_epoch = 0, 0

    # resume?
    if args.resume_from_checkpoint:
        ckpt_name = (
            os.path.basename(args.resume_from_checkpoint)
            if args.resume_from_checkpoint != "latest"
            else max((p for p in os.listdir(args.output_dir) if p.startswith("checkpoint-")), default=None)
        )
        if ckpt_name:
            ckpt_dir = Path(args.output_dir) / ckpt_name

            # <<< add this guard
            saved_mode = read_mode_marker(ckpt_dir)
            if saved_mode is not None and saved_mode != args.train_mode:
                raise RuntimeError(
                    f"Mismatched training mode for resume: checkpoint '{ckpt_name}' was saved as '{saved_mode}', "
                    f"but current run is '{args.train_mode}'.\n"
                    f"Use a matching checkpoint or switch --train_mode."
                )

            accelerator.load_state(str(ckpt_dir))
            global_step = int(ckpt_name.split("-")[1])
            first_epoch = global_step // steps_per_epoch
            logger.info(f"Resumed from {ckpt_name}")


    prog = tqdm(range(global_step, max_steps), disable=not accelerator.is_local_main_process, desc="Steps")

    # precompute constants
    vae_shift, vae_scale, vae_ch = vae.config.shift_factor, vae.config.scaling_factor, vae.config.block_out_channels

    # ------------------------------------------------------------------
    # helper to map timesteps → sigma
    def sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sched_sig = noise_sched_copy.sigmas.to(accelerator.device, dtype)
        sched_ts = noise_sched_copy.timesteps.to(accelerator.device)
        idx = [(sched_ts == t).nonzero().item() for t in timesteps]
        s = sched_sig[idx].flatten()
        while s.ndim < n_dim:
            s = s.unsqueeze(-1)
        return s

    # ------------------------------------------------------------------
    # Training loop -----------------------------------------------------
    # ------------------------------------------------------------------


    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(dl_train):
            with accelerator.accumulate(transformer):
                prompts = batch["prompts"]
                bsz = len(prompts)

                prompt_emb, pooled_emb, txt_ids = batch_prompt_emb(prompts, bsz)

                # latents
                pix = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)
                z0 = vae.encode(pix).latent_dist.sample()
                z0 = (z0 - vae_shift) * vae_scale
                z0 = z0.to(weight_dtype)

                # masks
                mask = batch["mask_pixel_values"].to(accelerator.device, dtype=vae.dtype)
                masked = pix * (1 - mask)
                m_lat, m_img_lat = prepare_mask_latents(
                    vae,
                    mask,
                    masked,
                    bsz,
                    16,
                    1,
                    pix.shape[2],
                    pix.shape[3],
                    z0.dtype,
                    accelerator.device,
                )
                m_img_lat = torch.cat([m_img_lat, m_lat], dim=-1)

                # flow‑matching noise
                noise = torch.randn_like(z0)
                u = compute_density_for_timestep_sampling(
                    args.weighting_scheme, bsz, args.logit_mean, args.logit_std, args.mode_scale
                )
                idx = (u * noise_sched_copy.config.num_train_timesteps).long()
                ts = noise_sched_copy.timesteps[idx].to(z0.device)
                sigma = sigmas(ts, n_dim=z0.ndim, dtype=z0.dtype)
                zt = (1 - sigma) * z0 + sigma * noise

                packed = FluxFillPipeline._pack_latents(zt, bsz, zt.shape[1], zt.shape[2], zt.shape[3])
                latent_ids = FluxFillPipeline._prepare_latent_image_ids(
                    bsz, zt.shape[2] // 2, zt.shape[3] // 2, accelerator.device, weight_dtype
                )

                
                guidance = None
                if getattr(transformer.config, "guidance_embeds", False):
                    guidance = torch.full(
                        (bsz,),
                        args.guidance_scale,
                        device=accelerator.device,
                        dtype=weight_dtype,  
                    )

                pred = transformer(
                    hidden_states=torch.cat([packed, m_img_lat], dim=2),
                    timestep=ts / 1000.0,
                    guidance=guidance,
                    pooled_projections=pooled_emb,
                    encoder_hidden_states=prompt_emb,
                    txt_ids = txt_ids.reshape(-1, 3), 
                    img_ids=latent_ids,
                    return_dict=False,
                )[0]

                pred = FluxFillPipeline._unpack_latents(
                    pred,
                    height=zt.shape[2] * (2 ** (len(vae_ch) - 1)),
                    width=zt.shape[3] * (2 ** (len(vae_ch) - 1)),
                    vae_scale_factor=2 ** (len(vae_ch) - 1),
                )

                w = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigma)
                target = noise - z0
                loss = ((w * (pred - target) ** 2).reshape(bsz, -1).mean())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step(); lr_sched.step(); optimizer.zero_grad()

            # bookkeeping ------------------------------------------------
            if accelerator.sync_gradients:
                global_step += 1
                prog.update(1)

                # checkpointing every N steps --------------------------
                if (
                    accelerator.is_main_process
                    and global_step % args.checkpointing_steps == 0
                ):
                    # rotate old ckpts
                    if args.checkpoints_total_limit is not None:
                        ckpts = sorted(
                            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")],
                            key=lambda x: int(x.split("-")[1]),
                        )
                        while len(ckpts) >= args.checkpoints_total_limit:
                            rm = ckpts.pop(0)
                            shutil.rmtree(Path(args.output_dir) / rm)
                            logger.info(f"Removed old checkpoint {rm}")

                    path = Path(args.output_dir) / f"checkpoint-{global_step}"
                    accelerator.save_state(path)
                    logger.info(f"Saved checkpoint to {path}")

                # scalar logs ----------------------------------------
                accelerator.log({"loss": loss.detach().item(), "lr": lr_sched.get_last_lr()[0]}, step=global_step)
                prog.set_postfix(loss=f"{loss.detach().item():.4f}")

                # mid‑training validation ---------------------------
                if (
                    accelerator.is_main_process
                    and global_step % args.validation_steps == 0
                    and dl_val is not None
                ):
                    pipe = FluxFillPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        transformer=accelerator.unwrap_model(transformer),  # live LoRA if applicable
                        vae=vae,
                        torch_dtype=weight_dtype,
                    )
                    # NOTE: *no* load_lora_weights here — in‑memory is newest & fastest
                    log_validation(pipe, args, accelerator, dl_val, tag="validation")
                    del pipe; free_memory()

                if global_step >= max_steps:
                    break

        if global_step >= max_steps:
            break

    # ------------------------------------------------------------------
    # Final save + final validation ------------------------------------
    # ------------------------------------------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.train_mode == "lora":
            lora_state = get_peft_model_state_dict(unwrap(transformer).to(weight_dtype))
            FluxFillPipeline.save_lora_weights(args.output_dir, transformer_lora_layers=lora_state)
        else:
            unwrap(transformer).save_pretrained(os.path.join(args.output_dir, "transformer"))

        write_mode_marker(args.output_dir, args.train_mode)

        if dl_val is not None:
            save_dir = Path(args.output_dir) / "val"

            if args.train_mode == "lora":
                pipe = FluxFillPipeline.from_pretrained(
                    args.pretrained_model_name_or_path, torch_dtype=weight_dtype
                )
                pipe.load_lora_weights(args.output_dir)
            else:
                # Load the freshly saved base transformer
                tuned_tx = FluxTransformer2DModel.from_pretrained(
                    args.output_dir, subfolder="transformer", torch_dtype=weight_dtype
                )
                pipe = FluxFillPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    transformer=tuned_tx,
                    torch_dtype=weight_dtype,
                )
            log_validation(pipe, args, accelerator, dl_val, tag="test", save_dir=save_dir)
            del pipe; free_memory()

        if args.push_to_hub:
            from huggingface_hub import upload_folder

            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
