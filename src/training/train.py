import torch
from .evaluate import evaluate

def train_one_epoch(model, dataloader, lossfn, optimizer, device):
    model.train()
    current_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()               
        outputs = model(inputs) # ForwardProp
        loss = lossfn(outputs, labels) # Compute loss
        loss.backward() # BackProp
        optimizer.step() # Update weights

        current_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

    epoch_loss = current_loss / total_samples
    epoch_acc = 100.0 * total_correct / total_samples
    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, lossfn, optimizer, device, num_epochs):
    """Main driver for the training experiment."""
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        t_loss, t_acc = train_one_epoch(model, train_loader, lossfn, optimizer, device)
    
        v_loss, v_acc = evaluate(model, val_loader, lossfn, device)

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] -> "
              f"Train: {t_loss:.4f} ({t_acc:.2f}%) | "
              f"Val: {v_loss:.4f} ({v_acc:.2f}%)")

    return model, history ## Track in 'history'