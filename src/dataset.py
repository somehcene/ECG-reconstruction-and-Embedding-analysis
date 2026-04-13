import os
import ast
import numpy as np
import pandas as pd
import wfdb
import torch

from pathlib import Path
from torch.utils.data import Dataset


# PATHS


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
METADATA_PATH = PROJECT_ROOT / "results" / "ptbxl_metadata_with_superclasses.csv"



# PREPROCESSING


def normalize_per_lead(signal, eps=1e-8):
    """Z-score normalization per lead"""
    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True)
    return (signal - mean) / (std + eps)


def pad_or_truncate(signal, target_length=1000):
    """Ensure fixed-length ECG"""
    length = signal.shape[1]

    if length < target_length:
        pad_width = target_length - length
        signal = np.pad(signal, ((0, 0), (0, pad_width)), mode='constant')
    else:
        signal = signal[:, :target_length]

    return signal


def preprocess_signal(signal):
    """Full preprocessing pipeline"""
    signal = pad_or_truncate(signal)
    signal = normalize_per_lead(signal)
    return signal



# LOADING ECG


def load_ecg(path):
    signal, _ = wfdb.rdsamp(str(path))
    return signal.T.astype(np.float32)



# DATASET CLASS


class PTBXLDataset(Dataset):
    def __init__(self, df, label_cols, data_dir):
        self.df = df.reset_index(drop=True)
        self.label_cols = label_cols
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load signal
        path = self.data_dir / row["filename_lr"]
        signal = load_ecg(path)

        # Preprocess
        signal = preprocess_signal(signal)

        # Labels
        labels = row[self.label_cols].values.astype(np.float32)

        return torch.tensor(signal), torch.tensor(labels)



# DATA LOADING 


def load_data(selected_classes=None):
    """
    Load dataset with flexible class selection

    selected_classes:
        None → use all 5 classes
        list → subset (e.g. ["MI","STTC","CD"])
    """

    df = pd.read_csv(METADATA_PATH)

    # Parse scp_codes 
    if isinstance(df["scp_codes"].iloc[0], str):
        df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    all_classes = ["CD", "HYP", "MI", "NORM", "STTC"]

    label_cols = selected_classes if selected_classes else all_classes

    # Official PTB-XL split
    train_df = df[df["strat_fold"].between(1, 8)].copy()
    val_df = df[df["strat_fold"] == 9].copy()
    test_df = df[df["strat_fold"] == 10].copy()

    return train_df, val_df, test_df, label_cols



# CLASS WEIGHTS (when using all 5 superclasses)


def compute_class_weights(df, label_cols):
    """
    Compute pos_weight for BCEWithLogitsLoss

    Formula (standard):
        pos_weight = (#negative / #positive)

    This is recommended for imbalanced multi-label tasks
    """

    labels = df[label_cols].values

    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts

    pos_weight = neg_counts / (pos_counts + 1e-6)

    return torch.tensor(pos_weight, dtype=torch.float32)



# DATALOADERS

def create_dataloaders(batch_size=32, selected_classes=None):

    from torch.utils.data import DataLoader

    train_df, val_df, test_df, label_cols = load_data(selected_classes)

    train_dataset = PTBXLDataset(train_df, label_cols, DATA_DIR)
    val_dataset = PTBXLDataset(val_df, label_cols, DATA_DIR)
    test_dataset = PTBXLDataset(test_df, label_cols, DATA_DIR)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    # compute class weights ONLY on train set
    class_weights = compute_class_weights(train_df, label_cols)

    return train_loader, val_loader, test_loader, label_cols, class_weights