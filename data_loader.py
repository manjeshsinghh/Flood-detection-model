"""
Data loading and preprocessing utilities for flood classification.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from typing import Tuple, Optional


class FloodDataset(Dataset):
    #Dataset class for flood-prone and non-flood-prone image classification.
    
    def __init__(self, flood_dir: str, non_flood_dir: str, transform: Optional[transforms.Compose] = None):
        
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load flood-prone images (label = 1)
        if os.path.exists(flood_dir):
            for filename in os.listdir(flood_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.images.append(os.path.join(flood_dir, filename))
                    self.labels.append(1)
        
        # Load non-flood-prone images (label = 0)
        if os.path.exists(non_flood_dir):
            for filename in os.listdir(non_flood_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.images.append(os.path.join(non_flood_dir, filename))
                    self.labels.append(0)
        
        total_flood = sum(self.labels)
        total_non_flood = len(self.labels) - total_flood
        print(f"Loaded {total_flood} flood-prone images and {total_non_flood} non-flood-prone images")
        if total_flood == 0:
            print(f"WARNING: No flood-prone images found in {flood_dir}")
        if total_non_flood == 0:
            print(f"WARNING: No non-flood-prone images found in {non_flood_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """Get an image and its label."""
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


class TransformedDataset(Dataset):
    
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)


def get_transforms(image_size: int = 224, augment: bool = True):
    
    if augment:
        # Training transforms with data augmentation
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/test transforms without augmentation
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_data_loaders(
    flood_dir: str,
    non_flood_dir: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    image_size: int = 224,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    # Create full dataset
    full_dataset = FloodDataset(
        flood_dir=flood_dir,
        non_flood_dir=non_flood_dir,
        transform=None  # Will apply transforms separately
    )
    
    # Split dataset
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create wrapper datasets with transforms
    train_dataset = TransformedDataset(train_dataset, get_transforms(image_size, augment=True))
    val_dataset = TransformedDataset(val_dataset, get_transforms(image_size, augment=False))
    test_dataset = TransformedDataset(test_dataset, get_transforms(image_size, augment=False))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

