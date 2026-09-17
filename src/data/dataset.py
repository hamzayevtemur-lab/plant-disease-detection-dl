import os
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms() -> transforms.Compose:
    """Returns training data augmentation transforms."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms() -> transforms.Compose:
    """Returns evaluation/validation transforms (no data augmentation)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class PlantVillageDataset(Dataset):
    """PyTorch Dataset for PlantVillage crop disease images."""
    def __init__(self, csv_file: str, dataset_dir: str = "data/raw/plantvillage", transform: Optional[transforms.Compose] = None):
        import pandas as pd
        from datasets import load_from_disk
        self.data = pd.read_csv(csv_file)
        self.dataset = load_from_disk(dataset_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.data.iloc[index]
        image_index = int(row["image_index"])
        label = int(row["class_idx"])

        image = self.dataset[image_index]["image"]
        if self.transform:
            image = self.transform(image)

        return image, label


def create_dataloaders(
    splits_dir: str = "data/splits",
    dataset_dir: str = "data/raw/plantvillage",
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates train, validation, and test PyTorch DataLoaders."""
    train_dataset = PlantVillageDataset(
        csv_file=os.path.join(splits_dir, "train.csv"),
        dataset_dir=dataset_dir,
        transform=get_train_transforms()
    )

    validation_dataset = PlantVillageDataset(
        csv_file=os.path.join(splits_dir, "validation.csv"),
        dataset_dir=dataset_dir,
        transform=get_eval_transforms()
    )

    test_dataset = PlantVillageDataset(
        csv_file=os.path.join(splits_dir, "test.csv"),
        dataset_dir=dataset_dir,
        transform=get_eval_transforms()
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, validation_loader, test_loader