from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class TwoViewTransform:
    """Returns two augmented views of the same image.

    Pass a single transform for symmetric augmentation (SimCLR),
    or two different transforms for asymmetric augmentation (BYOL).
    """

    def __init__(self, transform1: transforms.Compose, transform2: transforms.Compose | None = None):
        self.transform1 = transform1
        self.transform2 = transform2 if transform2 is not None else transform1

    def __call__(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform1(x), self.transform2(x)


def simclr_augmentation(image_size: int, normalize: transforms.Normalize) -> transforms.Compose:
    """Exact SimCLR augmentation pipeline (Chen et al., 2020 — Table 1).

    Parameters match the paper exactly:
    - RandomResizedCrop  scale=(0.08, 1.0)
    - ColorJitter        strength s=1: (0.8s, 0.8s, 0.8s, 0.2s) at p=0.8
    - RandomGrayscale    p=0.2
    - GaussianBlur       kernel=10% of image size, sigma~U(0.1,2.0), p=0.5
    - RandomHorizontalFlip p=0.5
    """
    ks = int(0.1 * image_size)
    if ks % 2 == 0:
        ks += 1  # GaussianBlur requires an odd kernel size

    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=ks, sigma=(0.1, 2.0))
        ], p=0.5),
        transforms.ToTensor(),
        normalize,
    ])


def byol_augmentation(
    image_size: int, normalize: transforms.Normalize
) -> tuple[transforms.Compose, transforms.Compose]:
    """Exact BYOL augmentation pipeline (Grill et al., 2020 — Appendix B).

    Returns (view1, view2) — the two branches are asymmetric:

    view1 (online branch):
    - GaussianBlur  p=1.0  (always applied)
    - Solarize      p=0.0  (never applied)

    view2 (target branch):
    - GaussianBlur  p=0.1  (rarely applied)
    - Solarize      p=0.2

    Everything else is identical to SimCLR (same crop scale, jitter, grayscale).
    """
    ks = int(0.1 * image_size)
    if ks % 2 == 0:
        ks += 1

    shared = [
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ]

    view1 = transforms.Compose([
        *shared,
        transforms.GaussianBlur(kernel_size=ks, sigma=(0.1, 2.0)),   # p=1.0
        transforms.ToTensor(),
        normalize,
    ])

    view2 = transforms.Compose([
        *shared,
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=ks, sigma=(0.1, 2.0))
        ], p=0.1),
        transforms.RandomSolarize(threshold=128, p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    return view1, view2


def _build_ssl_transform(
    augmentation: str, image_size: int, normalize: transforms.Normalize
) -> tuple[transforms.Compose, transforms.Compose]:
    """Dispatch augmentation name to a (view1, view2) transform pair.

    SimCLR returns the same transform object twice — TwoViewTransform applies
    it independently each call, producing two different random augmentations.
    """
    if augmentation == "simclr":
        t = simclr_augmentation(image_size, normalize)
        return t, t
    elif augmentation == "byol":
        return byol_augmentation(image_size, normalize)
    raise ValueError(f"Unknown augmentation '{augmentation}'. Choose 'simclr' or 'byol'.")


def _clean_transform(image_size: int, normalize: transforms.Normalize) -> transforms.Compose:
    """Deterministic eval transform: resize → center crop → normalize."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])


# ---------------------------------------------------------------------------
# SSL Dataset wrapper
# ---------------------------------------------------------------------------

class SSLDataset(Dataset):
    """Wraps any (image, label) dataset to produce two augmented views per sample.

    The underlying dataset should be constructed with transform=None so it
    returns raw PIL images. SSLDataset then applies TwoViewTransform internally,
    yielding (view1, view2, label) tuples that collate correctly into
    (B, C, H, W), (B, C, H, W), (B,) batches with a standard DataLoader.

    Example — SimCLR on CIFAR-100:
        normalize = transforms.Normalize(...)
        base_ds = datasets.CIFAR100(root="./data", train=True, download=True, transform=None)
        ssl_ds  = SSLDataset(base_ds, simclr_augmentation(32, normalize))
        loader  = DataLoader(ssl_ds, batch_size=256, shuffle=True)

    Example — BYOL on CIFAR-100:
        view1, view2 = byol_augmentation(32, normalize)
        ssl_ds = SSLDataset(base_ds, view1, view2)
    """

    def __init__(
        self,
        dataset: Dataset,
        transform1: transforms.Compose,
        transform2: transforms.Compose | None = None,
    ):
        self.dataset = dataset
        self.two_view = TwoViewTransform(transform1, transform2)

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        img, label = self.dataset[idx]
        view1, view2 = self.two_view(img)
        return view1, view2, label


# ---------------------------------------------------------------------------
# TinyImageNet (not in torchvision — loaded via HuggingFace datasets)
# ---------------------------------------------------------------------------

class TinyImageNetDataset(Dataset):
    """Thin wrapper around the HuggingFace 'Maysee/tiny-imagenet' dataset."""

    def __init__(self, split: str, transform=None):
        from datasets import load_dataset  # lazy import — optional dependency
        hf = load_dataset("Maysee/tiny-imagenet", split=split)
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

CIFAR100_NORMALIZE = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
IMAGENET_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],    std=[0.229, 0.224, 0.225])


def _make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool = False) -> DataLoader:
    pin = torch.cuda.is_available()
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0,
    )


def get_cifar100_loaders(
    batch_size: int,
    data_root: str = "./data",
    num_workers: int = 2,
    augmentation: str = "simclr",
) -> tuple[DataLoader, DataLoader]:
    """Returns (train_loader, val_loader) for CIFAR-100.

    train_loader yields (view1, view2, label) — two SSL views per image.
    val_loader   yields (image, label) with deterministic preprocessing.

    Args:
        augmentation: "simclr" or "byol"
    """
    root = Path(data_root)
    aug = _build_ssl_transform(augmentation, 32, CIFAR100_NORMALIZE)

    base_train = datasets.CIFAR100(root=root, train=True,  download=True, transform=None)
    base_val   = datasets.CIFAR100(root=root, train=False, download=True, transform=None)

    train_ds = SSLDataset(base_train, *aug)
    base_val.transform = _clean_transform(32, CIFAR100_NORMALIZE)

    return (
        _make_dataloader(train_ds, batch_size, shuffle=True,  num_workers=num_workers, drop_last=True),
        _make_dataloader(base_val, batch_size, shuffle=False, num_workers=num_workers),
    )


def get_tinyimagenet_loaders(
    batch_size: int,
    num_workers: int = 2,
    augmentation: str = "simclr",
) -> tuple[DataLoader, DataLoader]:
    """Returns (train_loader, val_loader) for TinyImageNet (200 classes, 64x64).

    Requires: pip install datasets
    Downloads and caches via HuggingFace on first call.

    train_loader yields (view1, view2, label) — two SSL views per image.
    val_loader   yields (image, label) with deterministic preprocessing.

    Args:
        augmentation: "simclr" or "byol"
    """
    aug = _build_ssl_transform(augmentation, 64, IMAGENET_NORMALIZE)

    base_train = TinyImageNetDataset(split="train")
    base_val   = TinyImageNetDataset(split="valid")

    train_ds = SSLDataset(base_train, *aug)
    base_val.transform = _clean_transform(64, IMAGENET_NORMALIZE)

    return (
        _make_dataloader(train_ds, batch_size, shuffle=True,  num_workers=num_workers, drop_last=True),
        _make_dataloader(base_val, batch_size, shuffle=False, num_workers=num_workers),
    )


def get_eval_loader(
    dataset: Dataset,
    batch_size: int,
    normalize: transforms.Normalize,
    image_size: int,
    num_workers: int = 2,
    augmented: bool = False,
    augmentation: str = "simclr",
) -> DataLoader:
    """Returns an eval DataLoader yielding (image, label).

    Args:
        dataset:      Any dataset returning (PIL image, label) with transform=None.
        normalize:    Normalization stats matching the dataset
                      (use CIFAR100_NORMALIZE or IMAGENET_NORMALIZE).
        image_size:   Target image size (32 for CIFAR-100, 64 for TinyImageNet).
        augmented:    False (default) — deterministic center-crop, for kNN / linear probe.
                      True — SSL augmentation applied, for test-time augmentation.
        augmentation: "simclr" or "byol" (only relevant when augmented=True).
    """
    if augmented:
        transform, _ = _build_ssl_transform(augmentation, image_size, normalize)  # view1 only
    else:
        transform = _clean_transform(image_size, normalize)

    dataset.transform = transform  # type: ignore
    return _make_dataloader(dataset, batch_size, shuffle=False, num_workers=num_workers)
