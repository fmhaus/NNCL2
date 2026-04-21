from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Normalization constants
# ---------------------------------------------------------------------------

CIFAR100_NORMALIZE = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
IMAGENET_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],    std=[0.229, 0.224, 0.225])


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _simclr_aug(image_size: int, normalize: transforms.Normalize) -> transforms.Compose:
    ks = int(0.1 * image_size)
    if ks % 2 == 0:
        ks += 1
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=ks, sigma=(0.1, 2.0))], p=0.5),
        transforms.ToTensor(),
        normalize,
    ])
    """

    return transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize
        ])


def _clean_transform(image_size: int, normalize: transforms.Normalize) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])


# ---------------------------------------------------------------------------
# Two-view dataset wrapper
# ---------------------------------------------------------------------------

class _TwoViewDataset(Dataset):
    """Wraps a dataset to yield (view1, view2, label) using two transforms."""

    def __init__(self, dataset: Dataset, t1: transforms.Compose, t2: transforms.Compose):
        self.dataset = dataset
        self.t1 = t1
        self.t2 = t2

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        img, label = self.dataset[idx]
        return self.t1(img), self.t2(img), label


# ---------------------------------------------------------------------------
# TinyImageNet
# ---------------------------------------------------------------------------

class TinyImageNetDataset(Dataset):
    """Thin wrapper around the HuggingFace 'Maysee/tiny-imagenet' dataset."""

    def __init__(self, split: str, transform=None):
        from datasets import load_dataset as hf_load
        hf = hf_load("Maysee/tiny-imagenet", split=split)
        hf = hf.select_columns(["image", "label"])
        self.data = hf
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        sample = self.data[idx]
        img = sample["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, sample["label"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(
    dataset: Literal["cifar100", "tinyimagenet"],
    two_view: bool,
    augment: Literal["simclr"] | None,
    train: bool,
    batch_size: int,
    num_workers: int = 4,
    data_root: str = "./data",
) -> DataLoader:
    """Single entry point for all data loading.

    Args:
        dataset:    "cifar100" or "tinyimagenet"
        two_view:   True  → yields (view1, view2, label) for SSL training
                    False → yields (image, label)
        augment:    "simclr" → apply SimCLR augmentation pipeline
                    None     → clean center-crop (eval)
        train:      True  → training split, shuffled, drop_last=True
                    False → val split, not shuffled
        batch_size: Batch size
        num_workers: DataLoader workers
        data_root:  Root directory for CIFAR-100 download
    """
    if dataset == "cifar100":
        image_size, normalize = 32, CIFAR100_NORMALIZE
    else:
        image_size, normalize = 64, IMAGENET_NORMALIZE

    if dataset == "cifar100":
        base = datasets.CIFAR100(root=data_root, train=train, download=True, transform=None)
    else:
        base = TinyImageNetDataset(split="train" if train else "valid")

    if augment and two_view:
        t = _simclr_aug(image_size, normalize)
        ds: Dataset = _TwoViewDataset(base, t, t)
    else:
        transform = _simclr_aug(image_size, normalize) if augment else _clean_transform(image_size, normalize)
        base.transform = transform  # type: ignore
        ds = base

    pin = torch.cuda.is_available()
    return DataLoader(
        ds, batch_size=batch_size, shuffle=train, drop_last=train,
        num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0,
    )
