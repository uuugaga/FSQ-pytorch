# Ignore UserWarning
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import lpips


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


def calculate_perceptual_similarity(folder1, folder2, batch_size=32, device=None):
    """
    Calculate Perceptual Similarity (LPIPS) between paired images in folder1 and folder2 using lpips library

    Args:
        folder1: Path to first folder of images
        folder2: Path to second folder of images
        batch_size: Batch size for processing
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Average LPIPS score across all image pairs
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define image transformations (LPIPS expects [-1, 1] range)
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Match common input size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            ),  # Scale to [-1, 1]
        ]
    )

    # Create datasets and dataloaders
    dataset1 = ImageFolderDataset(folder1, transform=preprocess)
    dataset2 = ImageFolderDataset(folder2, transform=preprocess)

    # Check if the number of images match
    if len(dataset1) != len(dataset2):
        raise ValueError(
            f"Number of images in folder1 ({len(dataset1)}) and folder2 ({len(dataset2)}) must match for LPIPS."
        )

    dataloader1 = DataLoader(
        dataset1, batch_size=batch_size, shuffle=False, num_workers=4
    )
    dataloader2 = DataLoader(
        dataset2, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Load LPIPS model (default is VGG16)
    lpips_model = lpips.LPIPS(net="vgg").to(device)
    lpips_model.eval()

    # Calculate LPIPS scores
    lpips_scores = []
    with torch.no_grad():
        for batch1, batch2 in zip(
            tqdm(
                dataloader1,
                desc="Processing folder1",
                unit="batch",
                ncols=75,
                leave=False,
            ),
            tqdm(
                dataloader2,
                desc="Processing folder2",
                unit="batch",
                ncols=75,
                leave=False,
            ),
        ):
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)

            # Compute LPIPS score for the batch
            score = lpips_model(batch1, batch2)  # Returns [batch_size]
            lpips_scores.append(score.cpu())

    # Concatenate and average all scores
    lpips_scores = torch.cat(lpips_scores, dim=0)
    lpips_score = lpips_scores.mean().item()

    return lpips_score


# Example usage
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Calculate Perceptual Similarity (LPIPS) between two folders of images"
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

    # Using LPIPS implementation
    lpips_score = calculate_perceptual_similarity(
        args.real, args.gen, batch_size=args.batch_size, device=args.device
    )
    print(f"\nLPIPS score: {lpips_score:.2f}")
