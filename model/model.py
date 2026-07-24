import torch
import torch.nn as nn
import torch.nn.functional as F

class EchoCRNN(nn.Module):
    def __init__(self, num_classes=8):
        super(EchoCRNN, self).__init__()
        
        # Input shape: (batch, 1, 64 mels, T frames)
        
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Block 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Block 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # After 3 MaxPool2d(2) layers, the frequency dimension (64 mels) is reduced by 2^3 = 8.
        # So 64 / 8 = 8 mel bins remain.
        # The channel dimension is 64.
        # When flattening frequency into channels, we get 64 channels * 8 bins = 512 features per time step.
        
        # GRU
        self.gru = nn.GRU(
            input_size=64 * 8, 
            hidden_size=64, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=False
        )
        
        # Classifier
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, 1, 64, T)
        
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # x is now (batch, 64, 8, T_reduced)
        
        # Reshape to (batch, T_reduced, features) for GRU
        batch_size, channels, freqs, frames = x.size()
        
        # Permute to (batch, frames, channels, freqs)
        x = x.permute(0, 3, 1, 2).contiguous() 
        
        # Flatten channels and freqs into a single feature dimension
        x = x.view(batch_size, frames, channels * freqs) # (batch, frames, 512)
        
        # GRU forward pass
        output, hn = self.gru(x)
        # hn shape is (1, batch, 64) because it's 1-layer, unidirectional
        
        # Extract the hidden state from the final time step
        gru_out = hn[0]
        
        # Fully connected layer
        logits = self.fc(gru_out)
        
        # Return raw logits. 
        # CrossEntropyLoss expects logits during training.
        return logits

    def predict(self, x):
        """Helper for inference returning probabilities"""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
