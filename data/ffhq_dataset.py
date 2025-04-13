import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from typing import Optional, Callable


class SingleFolderDataset(Dataset):
    """Dataset for loading all images from a folder without labels"""

    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        transform: Optional[Callable] = None,
        return_filenames: bool = False,
    ):

        self.data_dir = data_dir
        self.image_size = image_size
        self.return_filenames = return_filenames
        self.file_list = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".JPEG")
        ]

        print(f"Found {len(self.file_list)} images in {data_dir}")

        # Define default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.file_list[idx]
        try:
            image = Image.open(image_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            if self.return_filenames:
                return (image, image_path)
            else:
                return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a placeholder if image loading fails
            return torch.zeros(3, self.image_size, self.image_size)


def create_dataloader(
    data_dir: str,
    batch_size: int,
    image_size: int = 256,
    num_workers: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    pin_memory: bool = True,
    shuffle: bool = True,
    seed: int = 42,
    return_filenames: bool = False,
) -> DataLoader:

    # Create dataset
    dataset = SingleFolderDataset(
        data_dir=data_dir, image_size=image_size, return_filenames=return_filenames
    )

    # Setup sampler for distributed training
    sampler = None

    if distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed
        )
        shuffle = False  # Sampler handles shuffling

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return dataloader


# Example usage
if __name__ == "__main__":

    data_dir = "val2017"
    batch_size = 32
    image_size = 256

    dataloader = create_dataloader(data_dir, batch_size, image_size)
    for images in dataloader:
        print(
            images.shape
        )  # Should print torch.Size([batch_size, 3, image_size, image_size])
        break

    dataloader = create_dataloader(
        data_dir, batch_size, image_size, return_filenames=True
    )
    for images, filenames in dataloader:
        print(filenames)  # Should print the filenames of the images
        break
