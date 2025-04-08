import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = []

        # Get all image files from the folder
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        for file in os.listdir(folder_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                self.image_paths.append(os.path.join(folder_path, file))

        # Sort to ensure consistent ordering
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


def calculate_psnr(folder1, folder2, batch_size=32, device=None, max_value=1.0):
    """
    Calculate PSNR between paired images in folder1 and folder2

    Args:
        folder1: Path to first folder of images
        folder2: Path to second folder of images
        batch_size: Batch size for processing
        device: Device to use ('cuda' or 'cpu')
        max_value: Maximum pixel value of the images (default: 1.0 for [0, 1] range)

    Returns:
        Average PSNR score across all image pairs
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define image transformations
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # Standardize size
            transforms.ToTensor(),  # Convert to [0, 1] range
        ]
    )

    # Create datasets and dataloaders
    dataset1 = ImageFolderDataset(folder1, transform=transform)
    dataset2 = ImageFolderDataset(folder2, transform=transform)

    # Check if the number of images match
    if len(dataset1) != len(dataset2):
        raise ValueError(
            f"Number of images in folder1 ({len(dataset1)}) and folder2 ({len(dataset2)}) must match."
        )

    dataloader1 = DataLoader(
        dataset1, batch_size=batch_size, shuffle=False, num_workers=4
    )
    dataloader2 = DataLoader(
        dataset2, batch_size=batch_size, shuffle=False, num_workers=4
    )

    psnr_scores = []
    with torch.no_grad():
        for batch1, batch2 in zip(
            tqdm(
                dataloader1,
                desc="Processing batches",
                unit="batch",
                ncols=75,
                leave=False,
            ),
            dataloader2,
        ):
            batch1 = batch1.to(device)  # [B, C, H, W]
            batch2 = batch2.to(device)  # [B, C, H, W]

            # Calculate Mean Squared Error (MSE) for the batch
            mse = torch.mean((batch1 - batch2) ** 2, dim=(1, 2, 3))  # Mean over C, H, W

            # Calculate PSNR for each image in the batch
            psnr = 10 * torch.log10(
                max_value**2 / (mse + 1e-10)
            )  # Add small epsilon to avoid log(0)
            psnr_scores.append(psnr.cpu())

    # Concatenate and average all scores
    psnr_scores = torch.cat(psnr_scores, dim=0)
    psnr_score = psnr_scores.mean().item()

    return psnr_score


# Example usage
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Calculate PSNR between two folders of images"
    )
    argparser.add_argument(
        "--real", type=str, required=True, help="Path to first folder of images"
    )
    argparser.add_argument(
        "--gen", type=str, required=True, help="Path to second folder of images"
    )
    argparser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for processing"
    )
    argparser.add_argument(
        "--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu')"
    )
    args = argparser.parse_args()

    # Using PSNR implementation
    psnr_score = calculate_psnr(
        args.real, args.gen, batch_size=args.batch_size, device=args.device
    )
    print(f"PSNR score: {psnr_score:.2f}")
