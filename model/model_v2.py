"""
ECHO — Clean ML Model Architectures (Phase 9)
==============================================
Defines:
  1. EchoCNN  — Candidate A: Light 2D-CNN baseline (Clean, fast, mobile-friendly)
  2. EchoCRNN — Candidate B: CNN feature extractor + GRU temporal aggregator

Inputs: (batch, 1, 64 mel bins, T time frames)
Output: (batch, num_classes) raw logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=(2, 2)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=pool_size) if pool_size else nn.Identity()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        return x

class EchoCNN(nn.Module):
    """
    Candidate A: Small 2D CNN Baseline.
    Fast, efficient, minimal parameters (~70k).
    """
    def __init__(self, num_classes=6):
        super().__init__()
        self.block1 = ConvBlock(1, 16, pool_size=(2, 2))  # (64, T) -> (32, T//2)
        self.block2 = ConvBlock(16, 32, pool_size=(2, 2)) # (32, T//2) -> (16, T//4)
        self.block3 = ConvBlock(32, 64, pool_size=(2, 2)) # (16, T//4) -> (8, T//8)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # Input shape: (B, 1, 64, T)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x) # (B, 64, 1, 1)
        x = torch.flatten(x, 1) # (B, 64)
        logits = self.fc(x)     # (B, num_classes)
        return logits

class EchoCRNN(nn.Module):
    """
    Candidate B: CRNN Baseline (CNN + GRU).
    ConvBlocks extract spatial-spectral features, GRU models temporal dynamics.
    """
    def __init__(self, num_classes=6, rnn_hidden=128):
        super().__init__()
        self.block1 = ConvBlock(1, 16, pool_size=(2, 2)) # 64 -> 32
        self.block2 = ConvBlock(16, 32, pool_size=(2, 2)) # 32 -> 16
        self.block3 = ConvBlock(32, 64, pool_size=(2, 2)) # 16 -> 8

        # Frequency dimension reduced from 64 to 8 after 3 pooling layers of factor 2
        freq_dim_out = 8
        self.rnn_input_dim = 64 * freq_dim_out # 512

        self.gru = nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.fc = nn.Linear(rnn_hidden, num_classes)

    def forward(self, x):
        # Input shape: (B, 1, 64, T)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # Shape: (B, 64 channels, 8 freq_bins, T_reduced)

        B, C, F_bins, T_red = x.shape
        # Permute to (B, T_reduced, C * F_bins)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, T_red, C * F_bins)

        # GRU temporal modeling
        out, _ = self.gru(x) # (B, T_reduced, rnn_hidden)

        # Global temporal pooling (mean across time steps)
        out = torch.mean(out, dim=1) # (B, rnn_hidden)

        logits = self.fc(out) # (B, num_classes)
        return logits

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    print("Testing Models...")
    dummy_input = torch.randn(4, 1, 64, 63)

    cnn = EchoCNN(num_classes=6)
    out_cnn = cnn(dummy_input)
    print(f"EchoCNN output shape: {out_cnn.shape}, trainable params: {count_parameters(cnn):,}")
    assert out_cnn.shape == (4, 6)

    crnn = EchoCRNN(num_classes=6)
    out_crnn = crnn(dummy_input)
    print(f"EchoCRNN output shape: {out_crnn.shape}, trainable params: {count_parameters(crnn):,}")
    assert out_crnn.shape == (4, 6)

    print("Model verification PASSED!")
