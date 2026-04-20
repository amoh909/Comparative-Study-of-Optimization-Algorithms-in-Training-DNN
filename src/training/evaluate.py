import torch 

def evaluate(model, dataloader, lossfn, device):
    model.eval()
    current_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = lossfn(outputs, labels)

            batch_size = inputs.size(0)
            current_loss += loss.item() * batch_size
            
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

    avg_loss = current_loss / total_samples
    avg_acc = (total_correct / total_samples) * 100.0
    return avg_loss, avg_acc