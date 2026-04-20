import copy
from .evaluate import evaluate

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

        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    epoch_loss = current_loss / total_samples
    epoch_acc = (total_correct / total_samples) * 100.0

    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, lossfn, optimizer, device, num_epochs):
    """
    Main training loop with:
    - validation each epoch
    - best model saving (based on validation accuracy)
    - full history tracking
    """

    model.to(device)

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, lossfn, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, lossfn, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            best_indicator = " <--- Best Model Saved!"
        else:
            best_indicator = ""

        print(
            f"Epoch [{epoch+1}/{num_epochs}] -> "
            f"Train Loss: {train_loss:.4f} ({train_acc:.2f}%) | "
            f"Val Loss: {val_loss:.4f} ({val_acc:.2f}%)"
            f"{best_indicator}"
        )

    model.load_state_dict(best_model_wts)

    print("\nTraining Complete.")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")

    return model, history