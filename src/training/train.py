import torch
import copy
from .evaluate import evaluate
from .metrics import calculate_accuracy

def train_one_epoch(model, dataloader, lossfn, optimizer, device):
    model.train()
    current_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()               
        outputs = model(inputs)
        loss = lossfn(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        current_loss += loss.item() * batch_size
        
        acc_percentage = calculate_accuracy(outputs, labels)
        total_correct += (acc_percentage / 100.0) * batch_size
        total_samples += batch_size

    epoch_loss = current_loss / total_samples
    epoch_acc = (total_correct / total_samples) * 100.0
    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, lossfn, optimizer, device, num_epochs):
    """Main driver for the training experiment with Best Model Saving."""
    model.to(device)
    
    # Track the best performing version
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
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

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            best_indicator = " <--- Best Model Saved!"
        else:
            best_indicator = ""

        print(f"Epoch [{epoch+1}/{num_epochs}] -> "
              f"Train Loss: {t_loss:.4f} ({t_acc:.2f}%) | "
              f"Val Loss: {v_loss:.4f} ({v_acc:.2f}%){best_indicator}")

    model.load_state_dict(best_model_wts)
    print(f"\nTraining Complete. Best Val Accuracy: {best_val_acc:.2f}%")
    
    return model, history