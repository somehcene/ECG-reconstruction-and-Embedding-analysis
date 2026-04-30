import torch
import torch.nn as nn
import torch.nn.functional as F


class STN3d(nn.Module):
    """
    Input transform network for the first 3 ECG point coordinates:
    [time, lead, voltage]
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        """
        x: (B, 3, N)
        returns: (B, 3, 3)
        """
        batchsize = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=False)[0]   # (B, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # (B, 9)

        identity = torch.eye(3, device=x.device).view(1, 9).repeat(batchsize, 1)
        x = x + identity
        x = x.view(-1, 3, 3)

        return x


class STNkd(nn.Module):
    """
    Feature transform network on intermediate point features.
    """
    def __init__(self, k=64):
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        """
        x: (B, k, N)
        returns: (B, k, k)
        """
        batchsize = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=False)[0]   # (B, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # (B, k*k)

        identity = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(batchsize, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)

        return x


class PointNetFeatECG(nn.Module):
    """
    Full PointNet feature extractor adapted to ECG point sets.

    Input point format:
    [time, lead, voltage, embedding_1, ..., embedding_32]

    We apply the input transform only to the first 3 dimensions.
    """
    def __init__(self, input_dim=35, global_feat=True, feature_transform=True):
        super().__init__()

        self.input_dim = input_dim
        self.global_feat = global_feat
        self.feature_transform = feature_transform

        self.stn = STN3d()

        # Shared MLP over points
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        """
        x: (B, N, input_dim)
        returns:
            global features: (B, 1024) if global_feat=True
            trans_input: (B, 3, 3)
            trans_feat: (B, 64, 64) or None
        """
        B, N, D = x.shape
        assert D == self.input_dim, f"Expected input_dim={self.input_dim}, got {D}"

        # Split coordinates and auxiliary embedding features
        coords = x[:, :, :3]       # (B, N, 3)
        extra = x[:, :, 3:]        # (B, N, D-3)

        # PointNet expects (B, C, N)
        coords_t = coords.transpose(2, 1)  # (B, 3, N)

        # Input transform on [time, lead, voltage]
        trans_input = self.stn(coords_t)   # (B, 3, 3)

        coords_aligned = torch.bmm(coords, trans_input)  # (B, N, 3)

        # Concatenate aligned coords + unchanged extra features
        x = torch.cat([coords_aligned, extra], dim=2)    # (B, N, D)
        x = x.transpose(2, 1)                            # (B, D, N)

        # Shared MLP block 1
        x = F.relu(self.bn1(self.conv1(x)))              # (B, 64, N)

        # Feature transform on 64-d point features
        if self.feature_transform:
            trans_feat = self.fstn(x)                    # (B, 64, 64)
            x_t = x.transpose(2, 1)                     # (B, N, 64)
            x_t = torch.bmm(x_t, trans_feat)            # (B, N, 64)
            x = x_t.transpose(2, 1)                     # (B, 64, N)
        else:
            trans_feat = None

        pointfeat = x                                    # (B, 64, N)

        # Shared MLP block 2
        x = F.relu(self.bn2(self.conv2(x)))             # (B, 128, N)
        x = self.bn3(self.conv3(x))                     # (B, 1024, N)

        # Symmetric aggregation
        x = torch.max(x, 2, keepdim=False)[0]           # (B, 1024)

        if self.global_feat:
            return x, trans_input, trans_feat
        else:
            # segmentation-style output if ever needed
            x = x.view(B, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], dim=1), trans_input, trans_feat


class PointNetECGFull(nn.Module):
    """
    Full PointNet-style classifier for multi-label ECG classification.
    Outputs raw logits for BCEWithLogitsLoss.
    """
    def __init__(self, input_dim=35, num_classes=5, feature_transform=True):
        super().__init__()

        self.feature_transform = feature_transform
        self.feat = PointNetFeatECG(
            input_dim=input_dim,
            global_feat=True,
            feature_transform=feature_transform
        )

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=0.3)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        """
        x: (B, N, input_dim)
        returns:
            logits: (B, num_classes)
            trans_input: (B, 3, 3)
            trans_feat: (B, 64, 64) or None
        """
        x, trans_input, trans_feat = self.feat(x)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        logits = self.fc3(x)

        return logits, trans_input, trans_feat


def feature_transform_regularizer(trans):
    """
    Regularization term from PointNet:
    encourage transform matrix to be close to orthogonal.
    """
    if trans is None:
        return 0.0

    d = trans.size(1)
    batchsize = trans.size(0)
    I = torch.eye(d, device=trans.device).unsqueeze(0).repeat(batchsize, 1, 1)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


if __name__ == "__main__":
    B, N, D = 8, 12000, 35
    x = torch.randn(B, N, D)

    model = PointNetECGFull(input_dim=D, num_classes=5, feature_transform=True)
    logits, trans_input, trans_feat = model(x)

    print("logits:", logits.shape)          # (8, 5)
    print("trans_input:", trans_input.shape) # (8, 3, 3)
    if trans_feat is not None:
        print("trans_feat:", trans_feat.shape)  # (8, 64, 64)

    reg = feature_transform_regularizer(trans_feat)
    print("regularizer:", reg.item() if torch.is_tensor(reg) else reg)