import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# train for 1 epoch

def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Train model for one epoch

    Returns:
        avg_loss
    """

    model.train()
    total_loss = 0

    for signals, labels in tqdm(loader, desc="Training"):
        signals = signals.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(signals)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# validation

def evaluate(model, loader, criterion, device):
    """
    Evaluate model on validation/test set

    Returns:
        avg_loss, macro_auc
    """

    model.eval()
    total_loss = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for signals, labels in tqdm(loader, desc="Evaluating"):
            signals = signals.to(device)
            labels = labels.to(device)

            outputs = model(signals)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            probs = torch.sigmoid(outputs)

            y_true.append(labels.cpu().numpy())
            y_pred.append(probs.cpu().numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    try:
        macro_auc = roc_auc_score(y_true, y_pred, average="macro")
    except:
        macro_auc = 0.0  # safety if class missing

    return total_loss / len(loader), macro_auc


# training loop 

def train_model(
    model,
    train_loader,
    val_loader,
    class_weights,   
    epochs=10,
    lr=1e-3,
    device="cuda"
):
    """
    Full training loop

    Returns:
        trained model + history
    """

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss 
    # Standard for multi-label ECG classification
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))

    # Optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": []
    }

    best_auc = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # TRAIN 
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # VALIDATE 
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val AUC:    {val_auc:.4f}")

        # Save the best model 
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "best_model.pt")
            print("Best model saved")
        
        patience = 5
        counter = 0

        if val_auc > best_auc:
            best_auc = val_auc
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered")
            break 

    return model, history


# test 

def test_model(model, test_loader, label_cols, device="cuda"):
    """
    Final evaluation on test set
    """

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for signals, labels in tqdm(test_loader, desc="Testing"):
            signals = signals.to(device)
            labels = labels.to(device)

            outputs = model(signals)
            probs = torch.sigmoid(outputs)

            y_true.append(labels.cpu().numpy())
            y_pred.append(probs.cpu().numpy())

    from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    macro_auc = roc_auc_score(y_true, y_pred, average="macro")

    # binarisation
    y_pred_bin = (y_pred > 0.5).astype(int)

    print(f"\nTest Macro AUC: {macro_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_bin, target_names=label_cols))

    print("\nF1 (macro):", f1_score(y_true, y_pred_bin, average="macro"))
    print("Precision (macro):", precision_score(y_true, y_pred_bin, average="macro"))
    print("Recall (macro):", recall_score(y_true, y_pred_bin, average="macro"))

    return y_true, y_pred, macro_auc

    