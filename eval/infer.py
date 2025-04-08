import os
import sys
import argparse
import torch
from torchvision.utils import save_image
from tqdm import tqdm
import yaml

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from data import create_dataloader  # Import both Dataset and DataLoader utility
from model import FSQVAE


def load_config(yaml_path):
    try:
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"yaml file not found: {yaml_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"parse error in yaml file: {yaml_path}, error: {e}")


def get_default_config_path():
    parser = argparse.ArgumentParser(
        description="Generate reconstructed images using FSQVAE on a single GPU"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()
    return args.config


def setup_device():
    """Set up device (GPU or CPU)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def generate_recon_images():
    # Load configuration
    config_path = get_default_config_path()
    config = load_config(config_path)

    # Setup device
    device = setup_device()

    # Print device info
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Create output directory for reconstructed images
    output_dir = config["val"]["output_dir"]
    recon_dir = os.path.join(output_dir, "fsqvae_recon")
    os.makedirs(recon_dir, exist_ok=True)
    print(f"Reconstructed images will be saved to: {recon_dir}")

    # Create DataLoader
    batch_size = config.get("val", {}).get(
        "batch_size", 32
    )  # Default to 32 if not specified
    dataloader = create_dataloader(
        data_dir=config["val"]["data_dir"],
        batch_size=batch_size,
        image_size=config["model"]["image_size"],
        num_workers=config.get("val", {}).get("num_workers", 4),
        distributed=False,  # Single GPU, no distributed training
        shuffle=False,  # Preserve order for consistent naming
        return_filenames=True,  # Add this to your create_dataloader if needed
    )

    # Load FSQVAE model
    model = FSQVAE(levels=config["model"]["levels"]).to(device)

    # Load checkpoint
    checkpoint = torch.load(config["val"]["ckpt_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from: {config['val']['ckpt_path']}")
    print(f"Codebook size: {model.fsq.codebook_len}")

    # Process images using DataLoader
    with torch.no_grad():
        # Progress bar
        pbar = tqdm(dataloader, ncols=110, desc="Generating reconstructed images")
        total_processed = 0

        for batch in pbar:
            # Get images and filenames from batch
            images, filenames = batch
            images = images.to(device)

            # Encode and decode using FSQVAE
            recon_images, _ = model(images)
            recon_images = recon_images.clamp(-1, 1)

            # Rescale to [0, 1] for saving
            recon_images_rescaled = (recon_images + 1) / 2

            # Save reconstructed images
            for i in range(images.size(0)):
                original_filename = os.path.basename(filenames[i])
                save_image(recon_images_rescaled[i], f"{recon_dir}/{original_filename}")

            # Update progress
            total_processed += images.size(0)
            pbar.set_description(f"Processed {total_processed} images")

    print(f"All reconstructed images saved to: {recon_dir}")


if __name__ == "__main__":
    generate_recon_images()
