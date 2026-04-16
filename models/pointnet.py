import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetECG(nn.Module):
    def __init__(self, input_dim, num_classes=5):
        super().__init__()

        # shared MLP (point-wise)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        x: (B, N, input_dim)
        """

        # point-wise feature extraction
        x = self.mlp(x)  # (B, N, 256)

        # global max pooling
        x = torch.max(x, dim=1)[0]  # (B, 256)

        # classification
        out = self.classifier(x)

        return out