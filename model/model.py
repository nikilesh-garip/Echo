import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=2):
        super(ConvBlock, self).__init__()
        # ConvBlock consists of two consecutive Conv2D-BatchNorm-ELU layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.elu1 = nn.ELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.elu2 = nn.ELU()
        self.pool = nn.MaxPool2d(kernel_size=pool_size) if pool_size else nn.Identity()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.elu1(self.bn1(self.conv1(x)))
        x = self.elu2(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        return x

class EchoTransformer(nn.Module):
    def __init__(self, num_classes=8, d_model=128, nhead=4, num_layers=2):
        super(EchoTransformer, self).__init__()
        
        # 1. CNN Feature Extractor
        # Input shape: (batch, 1, 192 frequency bins, T time frames)
        # Note: 192 bins comes from the 3x concatenated spectrogram (Original + Sobel + Laplacian)
        self.conv1 = ConvBlock(1, 16, pool_size=2)  # 192 -> 96
        self.conv2 = ConvBlock(16, 32, pool_size=2) # 96 -> 48
        self.conv3 = ConvBlock(32, 64, pool_size=2) # 48 -> 24
        
        # After 3 MaxPool layers of kernel 2, frequency bins downsample by 8 (192 / 8 = 24).
        # Flattened features dimension per time step: 64 channels * 24 bins = 1536 features.
        self.feature_proj = nn.Linear(64 * 24, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model)) # Max sequence length 1000 frames
        
        # 3. Transformer Encoder (Self-Attention temporal model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.2,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Classifier
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Input: (batch, 1, 192, T)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # Shape: (batch, 64, 24, T_reduced)
        
        batch_size, channels, freqs, frames = x.size()
        
        # Permute to (batch, T_reduced, channels, freqs)
        x = x.permute(0, 3, 1, 2).contiguous()
        # Flatten to (batch, T_reduced, 1536)
        x = x.view(batch_size, frames, channels * freqs)
        
        # Project to d_model (batch, T_reduced, 128)
        x = self.feature_proj(x)
        
        # Add Positional Encoding
        x = x + self.pos_encoder[:, :frames, :]
        
        # Self-Attention encoding
        x = self.transformer_encoder(x)
        
        # Global Temporal Average Pooling to get clip-level embeddings
        x = torch.mean(x, dim=1) # (batch, d_model)
        
        # Classifier
        logits = self.fc(x)
        return logits

    def predict(self, x):
        """Helper for inference returning class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
