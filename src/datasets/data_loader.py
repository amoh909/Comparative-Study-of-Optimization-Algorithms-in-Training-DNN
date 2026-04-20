import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from .transforms import get_transforms

def get_dataloaders(dataset, batch_size = 64, val_split = 0.1, seed = 42):
    """
    Downloads data, splits training into train/val, and returns 3 DataLoaders.
    """
    transform = get_transforms(dataset)
    data_path = f"./data/{dataset}"

    if dataset == "fashion_mnist":
        full_train_set = datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transform)
    elif dataset == "cifar10":
        full_train_set = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    val_size = int(len(full_train_set) * val_split)
    train_size = len(full_train_set) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train_set, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader