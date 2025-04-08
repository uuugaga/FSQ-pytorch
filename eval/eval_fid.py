import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models import inception_v3, Inception_V3_Weights
import numpy as np
from scipy.linalg import sqrtm
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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


def calculate_fid(folder1, folder2, batch_size=32, device=None):
    """
    Calculate FID between images in folder1 and folder2 using manual feature extraction

    Args:
        folder1: Path to first folder of images
        folder2: Path to second folder of images
        batch_size: Batch size for processing
        device: Device to use ('cuda' or 'cpu')

    Returns:
        FID score
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define image transformations
    preprocess = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ]
    )

    # Create datasets and dataloaders
    dataset1 = ImageFolderDataset(folder1, transform=preprocess)
    dataset2 = ImageFolderDataset(folder2, transform=preprocess)

    dataloader1 = DataLoader(
        dataset1, batch_size=batch_size, shuffle=False, num_workers=4
    )
    dataloader2 = DataLoader(
        dataset2, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Load inception model
    inception = inception_v3(
        weights=Inception_V3_Weights.DEFAULT, transform_input=False
    )
    inception.fc = torch.nn.Identity()  # Remove the final classification layer
    inception.eval()
    inception.to(device)

    # Extract features from folder1
    features1 = []
    with torch.no_grad():
        for batch in tqdm(
            dataloader1,
            desc="Processing real images",
            unit="batch",
            ncols=75,
            leave=False,
        ):
            batch = batch.to(device)
            feat = inception(batch)
            features1.append(feat.cpu().numpy())
    features1 = np.concatenate(features1, axis=0)

    # Extract features from folder2
    features2 = []
    with torch.no_grad():
        for batch in tqdm(
            dataloader2,
            desc="Processing recon images",
            unit="batch",
            ncols=75,
            leave=False,
        ):
            batch = batch.to(device)
            feat = inception(batch)
            features2.append(feat.cpu().numpy())
    features2 = np.concatenate(features2, axis=0)

    # Calculate mean and covariance
    mu1, sigma1 = features1.mean(axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = features2.mean(axis=0), np.cov(features2, rowvar=False)

    # Calculate FID
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    # Check and correct imaginary numbers
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_score = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid_score


# Example usage
if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        description="Calculate FID score between two folders of images"
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

    # Using manual implementation
    fid_score_manual = calculate_fid(
        args.real, args.gen, batch_size=args.batch_size, device=args.device
    )
    print(f"FID score: {fid_score_manual:.2f}")
