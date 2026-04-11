import torchvision.transforms as transforms

def get_transforms(dataset):
    """
    Returns the appropriate transforms for Fashion-MNIST or CIFAR-10.
    """
    if dataset == "fashion_mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif dataset == "cifar10":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose 'fashion_mnist' or 'cifar10'.")