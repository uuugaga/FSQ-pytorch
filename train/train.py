import os
import sys
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
import datetime
import shutil

import bitsandbytes as bnb
import lpips

# Add project root to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from model import FSQVAE
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from data import FFHQDataset
from utils import load_config, WarmupScheduler


def parse_arguments():
    """Parse command line arguments for configuration path."""
    parser = argparse.ArgumentParser(description="Train a FSQVAE model with torchrun")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()
    return args.config


def setup_distributed():
    """Set up distributed training environment."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        # Initialize distributed backend
        torch.distributed.init_process_group(backend="nccl")
        n_gpu = 1

    return device, local_rank, n_gpu


def save_sample_reconstructions(model, images, output_path, use_amp, local_rank):
    """Save sample reconstructions for visualization."""
    with torch.no_grad():
        model.eval()
        sample_images = images[:8].cuda(local_rank)

        with autocast(device_type="cuda", enabled=use_amp):
            recon_images, _ = model(sample_images)

        # Denormalize images
        sample_images = sample_images * 0.5 + 0.5
        recon_images = recon_images * 0.5 + 0.5

        # Create comparison grid
        comparison = torch.cat([sample_images, recon_images], dim=0)
        save_image(comparison, output_path, nrow=8)

        model.train()


def main():
    """Main training function."""
    # Load configuration
    config_path = parse_arguments()
    config = load_config(config_path)

    # Configure mixed precision training
    use_amp = config["training"]["use_amp"]

    # Initialize distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    # Create gradient scalers for mixed precision training
    scaler_g = GradScaler(enabled=use_amp)
    scaler_d = GradScaler(enabled=use_amp)

    # Setup output directory structure with date
    if dist.get_rank() == 0:
        # Get dataset name and image size from config
        dataset_name = config["training"]["dataset_name"]
        image_size = config["model"]["image_size"]
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")

        # Create main output directory path
        output_dir = os.path.join(
            root_dir, "ckpts", f"{dataset_name}_{image_size}_{current_date}"
        )

        # Create subdirectories
        weights_dir = os.path.join(output_dir, "weights")
        samples_dir = os.path.join(output_dir, "samples")

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(weights_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)

        # Copy the original config file
        shutil.copy(
            config_path, os.path.join(output_dir, os.path.basename(config_path))
        )

    # Log environment information
    if dist.get_rank() == 0:
        print(f"World size: {dist.get_world_size()}")
        print(f"Using GPU: {torch.cuda.get_device_name(local_rank)}")
        print(
            f"GPU memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.2f} GB"
        )
        print(f"Mixed precision training: {'Enabled' if use_amp else 'Disabled'}")

    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    # Create output directories
    if dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)

    # Setup dataset and data loader
    dataset = FFHQDataset(config["training"]["data_dir"], config["model"]["image_size"])
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        sampler=sampler,
        num_workers=config["training"]["num_workers"],
        drop_last=True,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )

    # Build FSQVAE model
    model = FSQVAE(levels=config["model"]["levels"])

    # Build discriminator
    discriminator = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).cuda(
        local_rank
    )
    discriminator.classifier = nn.Sequential(
        discriminator.classifier[0], discriminator.classifier[1], nn.Linear(768, 2)
    ).cuda(local_rank)

    # Track codebook usage statistics
    if dist.get_rank() == 0:
        print(f"Codebook size: {model.fsq.codebook_len}")
        codebook_size = model.fsq.codebook_len
        codebook_usage = torch.zeros(codebook_size).cuda(local_rank)

    # Setup distributed model
    model.cuda(local_rank)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        broadcast_buffers=False,
    )

    # Setup perceptual loss
    lpips_loss_fn = lpips.LPIPS(net="vgg").cuda(local_rank)

    # Configure optimizer
    optimizer_name = config["optimizer"]["name"]
    optimizer_params = config["optimizer"].copy()
    optimizer_params.pop("name")

    optimizer_class = getattr(bnb.optim, optimizer_name, None)
    if optimizer_class is None:
        raise ValueError(f"Optimizer '{optimizer_name}' not found in bnb.optim")

    # Create optimizers
    optimizer_g = optimizer_class(model.parameters(), **optimizer_params)
    optimizer_d = optimizer_class(discriminator.parameters(), **optimizer_params)

    # Setup learning rate schedulers
    scheduler_g = None
    scheduler_d = None
    use_scheduler = config["training"]["use_scheduler"]

    if use_scheduler:
        total_steps = config["training"]["epochs"] * len(dataloader)

        scheduler_g = WarmupScheduler(
            optimizer_g,
            warmup_steps=config["training"]["warmup_steps"],
            total_steps=total_steps,
            min_lr=0.0,
            max_lr=config["optimizer"]["lr"],
            decay_type="cosine",
            decay_ratio=0.1,
        )

        scheduler_d = WarmupScheduler(
            optimizer_d,
            warmup_steps=config["training"]["warmup_steps"],
            total_steps=total_steps,
            min_lr=0.0,
            max_lr=config["optimizer"]["lr"],
            decay_type="cosine",
            decay_ratio=0.1,
        )

    # Training parameters
    global_step = 0
    epochs = config["training"]["epochs"]
    grad_accum_steps = config["training"]["gradient_accumulation_steps"]
    log_freq = config["logging"]["log_freq"]
    save_freq = config["logging"]["save_freq"]
    save_images_freq = config["logging"]["save_sample_freq"]

    # Loss function
    bce_loss = nn.BCEWithLogitsLoss()

    # Main training loop
    for epoch in range(epochs):
        # Set models to training mode
        model.train()
        discriminator.train()
        sampler.set_epoch(epoch)  # For different batch ordering each epoch

        # Initialize metrics tracking
        recon_losses = []
        lpips_losses = []
        g_adv_losses = []
        d_losses = []

        if dist.get_rank() == 0:
            codebook_usage.zero_()

        # Create progress bar on rank 0
        pbar = tqdm(dataloader, disable=dist.get_rank() != 0, ncols=140, leave=False)

        for batch_idx, images in enumerate(pbar):
            # Move images to GPU
            images = images.cuda(local_rank, non_blocking=True)

            # =================== Train Discriminator ===================
            optimizer_d.zero_grad()

            # Get reconstructed images without gradients for discriminator
            with torch.no_grad():
                with autocast(device_type="cuda", enabled=use_amp):
                    recon_images, _ = model(images)

            # Train discriminator with mixed precision
            with autocast(device_type="cuda", enabled=use_amp):
                # Real images should get positive labels (1)
                real_pred = discriminator(images)
                # Fake images should get negative labels (0)
                fake_pred = discriminator(recon_images.detach())

                # Real and fake labels with noise for label smoothing
                real_labels = torch.ones_like(real_pred).cuda(local_rank) - 0.1
                fake_labels = torch.zeros_like(fake_pred).cuda(local_rank) + 0.1

                # Discriminator loss
                d_real_loss = bce_loss(real_pred, real_labels)
                d_fake_loss = bce_loss(fake_pred, fake_labels)
                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss = d_loss / grad_accum_steps

            # Scale and backward with mixed precision
            scaler_d.scale(d_loss).backward()

            # Update weights if we've accumulated enough gradients
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(
                dataloader
            ):
                # Unscale before gradient clipping
                scaler_d.unscale_(optimizer_d)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                scaler_d.step(optimizer_d)
                scaler_d.update()

            d_losses.append(d_loss.item() * grad_accum_steps)

            # =================== Train Generator (VQVAE) ===================
            optimizer_g.zero_grad()

            # Forward pass with mixed precision
            with autocast(device_type="cuda", enabled=use_amp):
                # Forward pass
                recon_images, quant_indices = model(images)

                # Reconstruction losses
                recon_loss = F.mse_loss(recon_images, images)
                lpips_loss = lpips_loss_fn(recon_images, images).mean()

                # Adversarial loss - try to fool discriminator
                fake_pred = discriminator(recon_images)
                real_labels = torch.ones_like(fake_pred).cuda(local_rank) - 0.1
                g_adv_loss = bce_loss(fake_pred, real_labels)

                # Calculate adversarial weight with cosine annealing schedule
                current_adv_weight = config["training"]["adversarial_weight"][
                    0
                ] + config["training"]["adversarial_weight"][1] * (
                    1 - 0.5 * (1 + np.cos(np.pi * (epoch / epochs)))
                )

                # Combine losses with their weights
                total_loss = (
                    config["training"]["recon_weight"] * recon_loss
                    + config["training"]["lpips_weight"] * lpips_loss
                    + current_adv_weight * g_adv_loss
                )

                # Scale loss for gradient accumulation
                total_loss = total_loss / grad_accum_steps

            # Scale and backward with mixed precision
            scaler_g.scale(total_loss).backward()

            # Record loss metrics
            recon_losses.append(recon_loss.item())
            lpips_losses.append(lpips_loss.item())
            g_adv_losses.append(g_adv_loss.item())

            # Track codebook usage statistics
            if dist.get_rank() == 0:
                batch_usage = torch.zeros(codebook_size).cuda(local_rank)
                unique_indices = torch.unique(quant_indices)
                batch_usage[unique_indices] = 1
                codebook_usage += batch_usage

            # Update weights if we've accumulated enough gradients
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(
                dataloader
            ):
                # Unscale before gradient clipping
                scaler_g.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_g.step(optimizer_g)
                scaler_g.update()

                # Step schedulers if enabled
                if use_scheduler:
                    scheduler_g.step()
                    scheduler_d.step()

            # Update progress bar information
            if dist.get_rank() == 0 and (batch_idx + 1) % log_freq == 0:
                current_lr_g = optimizer_g.param_groups[0]["lr"]

                # Calculate recent metrics
                avg_recon = np.mean(recon_losses[-min(log_freq, len(recon_losses)) :])
                avg_perceptual = np.mean(
                    lpips_losses[-min(log_freq, len(lpips_losses)) :]
                )
                avg_g_adv = np.mean(g_adv_losses[-min(log_freq, len(g_adv_losses)) :])
                avg_d = np.mean(d_losses[-min(log_freq, len(d_losses)) :])

                # Calculate codebook usage statistics
                used_entries = torch.sum(codebook_usage > 0).item()
                usage_percentage = (used_entries / codebook_size) * 100

                # Update progress bar description
                pbar.set_description(f"Epoch [{epoch+1}/{epochs}]")
                desc_parts = [
                    f"[{epoch+1}/{epochs}]",
                    f"Recon: {avg_recon:.4f}",
                    f"LPIPS: {avg_perceptual:.4f}",
                    f"G: {avg_g_adv:.4f}",
                    f"D: {avg_d:.4f}",
                    f"adv_w: {current_adv_weight:.2f}",
                    f"LR: {current_lr_g:.6f}",
                    f"Codebook: {usage_percentage:.1f}%",
                ]
                pbar.set_description(" ".join(desc_parts))

            # Save sample reconstructions
            if dist.get_rank() == 0 and (global_step + 1) % save_images_freq == 0:
                output_path = f"{output_dir}/samples/recon_step_{global_step + 1}.png"
                save_sample_reconstructions(
                    model, images, output_path, use_amp, local_rank
                )

            global_step += 1

            # Save sample reconstructions
            if dist.get_rank() == 0 and (global_step + 1) % save_images_freq == 0:
                output_path = os.path.join(
                    samples_dir,
                    f"recon_step_{global_step + 1}.png",
                )
                save_sample_reconstructions(
                    model, images, output_path, use_amp, local_rank
                )

            global_step += 1

        # Save checkpoint at specified frequency
        if (epoch + 1) % save_freq == 0 and dist.get_rank() == 0:
            checkpoint_path = os.path.join(
                weights_dir, f"checkpoint_epoch_{epoch+1}.pt"
            )

            # Save model and training state
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.module.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_g_state_dict": optimizer_g.state_dict(),
                    "optimizer_d_state_dict": optimizer_d.state_dict(),
                    "scaler_g_state_dict": scaler_g.state_dict(),
                    "scaler_d_state_dict": scaler_d.state_dict(),
                    "global_step": global_step,
                    "use_amp": use_amp,
                },
                checkpoint_path,
            )

            # Also save latest model separately
            latest_path = os.path.join(weights_dir, "latest_checkpoint.pt")
            # Create symlink or copy the file
            if os.path.exists(latest_path):
                os.remove(latest_path)
            shutil.copy(checkpoint_path, latest_path)

        # Log epoch results
        if dist.get_rank() == 0:
            avg_recon = np.mean(recon_losses)
            avg_perceptual = np.mean(lpips_losses)
            avg_g_adv = np.mean(g_adv_losses)
            avg_d = np.mean(d_losses)
            used_entries = torch.sum(codebook_usage > 0).item()

            log_msg = [
                f"[{epoch+1}/{epochs}] ",
                f"Recon: {avg_recon:.4f},",
                f"LPIPS: {avg_perceptual:.4f},",
                f"G: {avg_g_adv:.4f},",
                f"D: {avg_d:.4f},",
                f"Codebook Usage: {(used_entries / codebook_size) * 100:.2f}%",
            ]

            print(" ".join(log_msg))


if __name__ == "__main__":
    main()
