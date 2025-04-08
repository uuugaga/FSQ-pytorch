import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse

from skimage.metrics import structural_similarity as ssim


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


def calculate_ssim(folder1, folder2, device=None):
    # Set device
    device = torch.device(
        device if device is not None and torch.cuda.is_available() else "cpu"
    )

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    dataset1 = ImageFolderDataset(folder1, transform=transform)
    dataset2 = ImageFolderDataset(folder2, transform=transform)

    if len(dataset1) != len(dataset2):
        raise ValueError(
            f"Number of images in folder1 ({len(dataset1)}) and folder2 ({len(dataset2)}) must match."
        )

    ssim_scores = []
    for i in tqdm(
        range(len(dataset1)), desc="Processing images", ncols=75, leave=False
    ):
        img1 = dataset1[i].numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        img2 = dataset2[i].numpy().transpose(1, 2, 0)
        score = ssim(
            img1, img2, multichannel=True, channel_axis=2, win_size=11, data_range=1.0
        )
        ssim_scores.append(score)

    return np.mean(ssim_scores)


# Example usage
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Calculate SSIM between two folders of images"
    )
    argparser.add_argument(
        "--real", type=str, required=True, help="Path to first folder of images"
    )
    argparser.add_argument(
        "--gen", type=str, required=True, help="Path to second folder of images"
    )
    argparser.add_argument(
        "--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu')"
    )
    args = argparser.parse_args()

    # Using SSIM implementation
    ssim_score = calculate_ssim(args.real, args.gen, device=args.device)
    print(f"SSIM score: {ssim_score:.2f}")
