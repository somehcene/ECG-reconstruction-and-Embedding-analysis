import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

from models.pointnet_full_ecg import PointNetECGFull, feature_transform_regularizer
from src.pointcloud_ecg import create_point_cloud

def _sample_random_input_lead(signals: torch.Tensor) -> torch.Tensor:
    lead_idx = torch.randint(0, signals.shape[1], (signals.shape[0],), device=signals.device)
    return torch.stack([signals[i, idx:idx + 1, :] for i, idx in enumerate(lead_idx)])


def evaluate_newpointnet(model, unet, loader, criterion, reg_weight=1e-3, device='cpu'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.eval(); unet.eval()
    total_loss = 0.0; y_true = []; y_pred = []
    with torch.no_grad():
        for signals, labels in tqdm(loader, desc='Evaluating Full PointNet'):
            signals = signals.to(device); labels = labels.to(device)
            input_lead = _sample_random_input_lead(signals)
            _, embeddings = unet(input_lead)
            point_cloud = create_point_cloud(signals, embeddings)
            logits, _, trans_feat = model(point_cloud)
            loss = criterion(logits, labels)
            if trans_feat is not None:
                loss = loss + reg_weight * feature_transform_regularizer(trans_feat)
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            y_true.append(labels.cpu().numpy())
            y_pred.append(probs.cpu().numpy())
    y_true = np.vstack(y_true); y_pred = np.vstack(y_pred)
    try:
        macro_auc = roc_auc_score(y_true, y_pred, average='macro')
    except ValueError:
        macro_auc = 0.0
    return total_loss / len(loader), macro_auc


def train_newpointnet(unet, train_loader, val_loader, num_classes=5, epochs=10, lr=1e-3, reg_weight=1e-3, device='cpu'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    unet = copy.deepcopy(unet).to(device)
    unet.eval()
    for p in unet.parameters():
        p.requires_grad = False
    input_dim = 3 + unet.embedding_dim
    model = PointNetECGFull(input_dim=input_dim, num_classes=num_classes, feature_transform=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    best_state = None; best_val_auc = -float('inf')
    for epoch in range(epochs):
        model.train(); total_loss = 0.0
        for signals, labels in tqdm(train_loader, desc=f'Training Full PointNet [{epoch+1}/{epochs}]'):
            signals = signals.to(device); labels = labels.to(device)
            input_lead = _sample_random_input_lead(signals)
            with torch.no_grad():
                _, embeddings = unet(input_lead)
            point_cloud = create_point_cloud(signals, embeddings)
            logits, _, trans_feat = model(point_cloud)
            loss = criterion(logits, labels)
            if trans_feat is not None:
                loss = loss + reg_weight * feature_transform_regularizer(trans_feat)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        val_loss, val_auc = evaluate_newpointnet(model, unet, val_loader, criterion, reg_weight=reg_weight, device=device)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss:   {val_loss:.4f}')
        print(f'Val AUC:    {val_auc:.4f}')
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            print('Best full PointNet model saved in memory')
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def test_newpointnet(model, unet, test_loader, label_cols, reg_weight=1e-3, device='cpu'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device); unet = unet.to(device)
    model.eval(); unet.eval()
    y_true = []; y_pred = []; total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for signals, labels in tqdm(test_loader, desc='Testing Full PointNet'):
            signals = signals.to(device); labels = labels.to(device)
            input_lead = _sample_random_input_lead(signals)
            _, embeddings = unet(input_lead)
            point_cloud = create_point_cloud(signals, embeddings)
            logits, _, trans_feat = model(point_cloud)
            loss = criterion(logits, labels)
            if trans_feat is not None:
                loss = loss + reg_weight * feature_transform_regularizer(trans_feat)
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            y_true.append(labels.cpu().numpy())
            y_pred.append(probs.cpu().numpy())
    y_true = np.vstack(y_true); y_pred = np.vstack(y_pred)
    macro_auc = roc_auc_score(y_true, y_pred, average='macro')
    y_pred_bin = (y_pred > 0.5).astype(int)
    print('\n===== FULL POINTNET TEST RESULTS =====')
    print(f'Test Loss: {total_loss / len(test_loader):.4f}')
    print(f'Macro AUC: {macro_auc:.4f}')
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred_bin, target_names=label_cols))
    print('\nF1 (macro):', f1_score(y_true, y_pred_bin, average='macro'))
    print('Precision (macro):', precision_score(y_true, y_pred_bin, average='macro'))
    print('Recall (macro):', recall_score(y_true, y_pred_bin, average='macro'))
    return y_true, y_pred, macro_auc
