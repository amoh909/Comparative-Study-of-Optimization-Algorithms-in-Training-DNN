import os
import json
import torch.nn as nn
import torch.optim as optim

from src.training.train import train_model
from src.training.evaluate import evaluate
from src.utils import get_device, set_seed


def run_optimizer_study(model_class, train_loader, val_loader, test_loader, dataset_name, epochs=10):
    """
    Runs a fair optimizer comparison study on a fixed model and dataset.

    Args:
        model_class: model class to instantiate (e.g., FMNISTCNN, CIFAR10CNN)
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        dataset_name: string used for saving files
        epochs: number of training epochs

    Returns:
        study_results: dictionary containing history and final test metrics for each optimizer
    """

    device = get_device()
    lossfn = nn.CrossEntropyLoss()

    save_path = os.path.join("results", "logs", dataset_name)
    os.makedirs(save_path, exist_ok=True)

    experiments = [
        {"name": "SGD", "opt_class": optim.SGD, "kwargs": {"lr": 0.01, "momentum": 0.9}},
        {"name": "Adagrad", "opt_class": optim.Adagrad, "kwargs": {"lr": 0.01}},
        {"name": "RMSprop", "opt_class": optim.RMSprop, "kwargs": {"lr": 0.001}},
        {"name": "Adam", "opt_class": optim.Adam, "kwargs": {"lr": 0.001}},
    ]

    study_results = {}

    for exp in experiments:
        print(f"\n>>> Running {exp['name']} on {dataset_name} <<<")

        set_seed(42)

        model = model_class().to(device)

        optimizer = exp["opt_class"](model.parameters(), **exp["kwargs"])

        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lossfn=lossfn,
            optimizer=optimizer,
            device=device,
            num_epochs=epochs
        )

        test_loss, test_acc = evaluate(model, test_loader, lossfn, device)

        study_results[exp["name"]] = {
            "history": history,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "optimizer_params": exp["kwargs"]
        }

        output_file = os.path.join(save_path, f"{exp['name']}_results.json")
        with open(output_file, "w") as f:
            json.dump(study_results[exp["name"]], f, indent=4)

        print(f"{exp['name']} Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

    return study_results