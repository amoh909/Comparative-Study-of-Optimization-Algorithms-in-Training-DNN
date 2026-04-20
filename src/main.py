from src.datasets.data_loader import get_dataloaders
from src.models.fmnist_cnn import FMNISTCNN
from src.models.cifar_cnn import CIFAR10CNN
from src.experiments.run_experiments import run_optimizer_study


def run_experiment(dataset_name):
    if dataset_name == "fashion_mnist":
        model_class = FMNISTCNN
    elif dataset_name == "cifar10":
        model_class = CIFAR10CNN
    else:
        raise ValueError("Unsupported dataset")

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=dataset_name,
        batch_size=64,
        val_split=0.1,
        seed=42
    )

    run_optimizer_study(
        model_class=model_class,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dataset_name=dataset_name,
        epochs=10
    )


def main():
    run_experiment("fashion_mnist")

    # run_experiment("cifar10")


if __name__ == "__main__":
    main()