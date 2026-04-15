import torch
import torch.nn as nn
import torch.nn.functional as F



# CONV BLOCK


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),

            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)



# DOWN BLOCK


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.conv(x)
        skip = x
        x = self.pool(x)
        return x, skip



# UP BLOCK


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        if x.size(-1) != skip.size(-1):
            diff = skip.size(-1) - x.size(-1)
            x = F.pad(x, (0, diff))

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)

        return x



# UNET 1D WITH THE EMBEDDINGS


class UNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=12, embedding_dim=32):
        super().__init__()
        self.out_channels = out_channels
        self.embedding_dim = embedding_dim

        # Encoder
        self.down1 = DownBlock(in_channels, 32)
        self.down2 = DownBlock(32, 64)
        self.down3 = DownBlock(64, 128)
        self.down4 = DownBlock(128, 256)

        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)

        # Decoder
        self.up4 = UpBlock(512 + 256, 256)
        self.up3 = UpBlock(256 + 128, 128)
        self.up2 = UpBlock(128 + 64, 64)
        self.up1 = UpBlock(64 + 32, 32)

        
        # separate features for each lead (12 * 32 = 384 channels)
        self.lead_features = nn.Conv1d(32, out_channels * embedding_dim, kernel_size=1)

        # groups=out_channels (groups=12).
        # the network is using only the 32 channels of the lead i to generate the signals (lead) i 
        self.signal_head = nn.Conv1d(
            in_channels = out_channels * embedding_dim, 
            out_channels = out_channels, 
            kernel_size = 1, 
            groups = out_channels
        )

    def forward(self, x):
        # Encoder & Decoder
        x, s1 = self.down1(x)
        x, s2 = self.down2(x)
        x, s3 = self.down3(x)
        x, s4 = self.down4(x)
        x = self.bottleneck(x) 
        x = self.up4(x, s4) 
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1) # Shape: (B, 32, T) 

        # features per lead 
        lead_feat = self.lead_features(x)  # Shape: (B, 12*d, T)
        
        # (1) signal reco (the actually supervised part)
        signals = self.signal_head(lead_feat) # Shape: (B, 12, T)

        # (2) Embeddings (self supervised by learning to reconstruct the signal)
        B, _, T = lead_feat.shape

        # we separate explicitly the dimension is the leads (12) and the embeddings (d)
        lead_feat_reshaped = lead_feat.view(B, self.out_channels, self.embedding_dim, T)
        
        # Temporal pooling (Global Average Pooling on the time axis)
        embeddings = torch.mean(lead_feat_reshaped, dim=-1) # Shape: (B, 12, d)

        return signals, embeddings