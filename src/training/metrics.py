import torch

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100.0 * correct / total

def get_device(): ## To run on GPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")