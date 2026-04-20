import torch 
from .metrics import calculate_accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    current_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = inputs.size(0)
            current_loss += loss.item() * batch_size
            
            acc_percentage = calculate_accuracy(outputs, labels)
            total_correct += (acc_percentage / 100.0) * batch_size
            total_samples += batch_size

    avg_loss = current_loss / total_samples
    avg_acc = (total_correct / total_samples) * 100.0
    return avg_loss, avg_acc