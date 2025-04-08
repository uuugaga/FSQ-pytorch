import os
import sys
import argparse
import torch
from diffusers import AutoencoderKL
from torchvision.utils import save_image
from tqdm import tqdm
import yaml

# Add root directory to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from data.ffhq_dataset import create_dataloader


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
        description="Generate reconstructed images using SD3 VAE on a single GPU"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()
    return args


def setup_device():
    """Set up device (GPU or CPU)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def generate_recon_images():
    # Load configuration
    args = get_default_config_path()
    config = load_config(args.config)

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
    recon_dir = os.path.join(output_dir, "sd3_recon")
    os.makedirs(recon_dir, exist_ok=True)
    print(f"Reconstructed images will be saved to: {recon_dir}")

    # Create DataLoader
    dataloader = create_dataloader(
        data_dir=config["val"]["data_dir"],
        batch_size=config["val"]["batch_size"],
        image_size=config["model"]["image_size"],
        num_workers=config["val"]["num_workers"],
        shuffle=False,
        return_filenames=True,
    )

    # Load SD3 VAE from Hugging Face
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae"
    ).to(device)
    vae.eval()

    # Process images in batches
    with torch.no_grad():
        # Progress bar
        pbar = tqdm(dataloader, ncols=110, desc="Generating reconstructed images")

        for batch_idx, (images, filenames) in enumerate(pbar):
            # Move images to device
            images = images.to(device)

            # Encode and decode using SD3 VAE
            latents = vae.encode(images).latent_dist.sample()
            recon_images = vae.decode(latents).sample
            recon_images = recon_images.clamp(-1, 1)

            # Rescale to [0, 1] for saving
            recon_images_rescaled = (recon_images + 1) / 2

            # Save each reconstructed image in the batch
            for i, (recon_image, filename) in enumerate(
                zip(recon_images_rescaled, filenames)
            ):
                original_filename = os.path.basename(filename)
                save_image(recon_image, f"{recon_dir}/{original_filename}")

            # Update progress bar
            pbar.set_description(f"Processed {(batch_idx + 1) * len(images)} images")

    print(f"All reconstructed images saved to: {recon_dir}")


if __name__ == "__main__":
    generate_recon_images()
