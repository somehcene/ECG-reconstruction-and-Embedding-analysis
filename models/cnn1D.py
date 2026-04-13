import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    Simple 1D CNN for multi-label ECG classification.

    Architecture:
    - 3 convolutional blocks
    - Global Average Pooling
    - Fully connected classifier

    Input shape: (batch_size, 12 leads, signal_length)
    """

    def __init__(self, num_classes):
        super().__init__()

        # Convolutional feature extractor 
        self.features = nn.Sequential(

            # Block 1
            nn.Conv1d(in_channels=12, out_channels=32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            # Global Average Pooling (reduces time dimension to 1)
            nn.AdaptiveAvgPool1d(1)
        )

        # Classification head 
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: tensor of shape (batch_size, 12, signal_length)

        Returns:
            logits (before sigmoid)
        """

        x = self.features(x)

        # Flatten: (batch, channels, 1) → (batch, channels)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x