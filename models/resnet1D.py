import torch
import torch.nn as nn


class ConvBlock(nn.Sequential):
    """
    Basic convolution block:
    Conv1D → BatchNorm → Activation
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation=True):
        padding = (kernel_size - 1) // 2

        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels)
        ]

        if activation:
            layers.append(nn.ReLU())

        super().__init__(*layers)


class ResidualBlock(nn.Module):
    """
    Residual block:
    - 2 convolution layers
    - Skip connection (identity or projection)

    Key idea:
    output = F(x) + x

    This helps gradient flow and allows deeper networks :contentReference[oaicite:0]{index=0}
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Main path
        self.conv = nn.Sequential(
            ConvBlock(in_channels, out_channels, stride=stride),
            ConvBlock(out_channels, out_channels, activation=False)
        )

        # Skip connection (identity or projection)
        if in_channels == out_channels and stride == 1:
            self.skip = nn.Identity()
        else:
            self.skip = ConvBlock(in_channels, out_channels,
                                  kernel_size=1, stride=stride, activation=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class ResNet1D(nn.Module):
    """
    ResNet1D for ECG classification (ResNet18 style)

    Architecture:
    - Initial conv layer (stem)
    - 4 stages of residual blocks
    - Global Average Pooling
    - Fully connected layer
    """

    def __init__(self, layers, num_classes, input_channels=12):
        """
        Args:
            layers: list defining number of blocks per stage (e.g. [2,2,2,2])
            num_classes: number of output classes
        """

        super().__init__()

        self.in_channels = 64

        #  Initial convolution 
        self.stem = nn.Sequential(
            ConvBlock(input_channels, 64, kernel_size=7, stride=2),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        #  Residual stages 
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        #  Classification head 
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        """
        Creates a stack of residual blocks
        """
        layers = []

        # First block may change dimensions (stride)
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = x.squeeze(-1)

        return self.fc(x)