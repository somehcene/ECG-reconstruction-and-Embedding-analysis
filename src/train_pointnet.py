import torch
import torch.nn as nn
from tqdm import tqdm

from models.pointnet import PointNetECG
from src.pointcloud_ecg import create_point_cloud


def train_pointnet(unet, train_loader, val_loader, epochs=10, lr=1e-3, device="cpu"):

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    unet.to(device)
    unet.eval()  

    input_dim = 3 + unet.embedding_dim * 1  # 3 + d

    model = PointNetECG(input_dim=input_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for signals, labels in tqdm(train_loader):

            signals = signals.to(device)
            labels = labels.to(device)

            lead_idx = torch.randint(0, 12, (signals.shape[0],), device=signals.device)

            input_lead = torch.stack([
                signals[i, idx:idx+1, :]
                for i, idx in enumerate(lead_idx)
            ])

            with torch.no_grad():
                _, embeddings = unet(input_lead)

            point_cloud = create_point_cloud(signals, embeddings)

            preds = model(point_cloud)

            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

        # validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for signals, labels in val_loader:

                signals = signals.to(device)
                labels = labels.to(device)

                lead_idx = torch.randint(0, 12, (signals.shape[0],), device=signals.device)

                input_lead = torch.stack([
                    signals[i, idx:idx+1, :]
                    for i, idx in enumerate(lead_idx)
                ])

                _, embeddings = unet(input_lead)

                point_cloud = create_point_cloud(signals, embeddings)

                preds = model(point_cloud)

                loss = criterion(preds, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

    return model

from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
import numpy as np


def test_pointnet(model, unet, test_loader, label_cols, device="cuda"):

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model.to(device)
    unet.to(device)

    model.eval()
    unet.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for signals, labels in tqdm(test_loader, desc="Testing PointNet"):

            signals = signals.to(device)
            labels = labels.to(device)

            lead_idx = torch.randint(0, 12, (signals.shape[0],), device=signals.device)

            input_lead = torch.stack([
                signals[i, idx:idx+1, :]
                for i, idx in enumerate(lead_idx)
            ])

            _, embeddings = unet(input_lead)

            point_cloud = create_point_cloud(signals, embeddings)

            outputs = model(point_cloud)
            probs = torch.sigmoid(outputs)

            y_true.append(labels.cpu().numpy())
            y_pred.append(probs.cpu().numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    macro_auc = roc_auc_score(y_true, y_pred, average="macro")

    y_pred_bin = (y_pred > 0.5).astype(int)

    print("\n===== POINTNET TEST RESULTS =====")
    print(f"Macro AUC: {macro_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_bin, target_names=label_cols))

    print("\nF1 (macro):", f1_score(y_true, y_pred_bin, average="macro"))
    print("Precision (macro):", precision_score(y_true, y_pred_bin, average="macro"))
    print("Recall (macro):", recall_score(y_true, y_pred_bin, average="macro"))

    return y_true, y_pred, macro_auc