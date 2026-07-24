import torch
from model import EchoCRNN

def test_crnn():
    print("Testing CRNN Architecture...")
    
    # Initialize model
    model = EchoCRNN(num_classes=8)
    
    # Create dummy tensor matching the output of dataset.py
    # (batch_size, channels, n_mels, T)
    # E.g., batch=2, channel=1, mels=64, T=63
    dummy_input = torch.randn(2, 1, 64, 63)
    
    print(f"Input Shape: {dummy_input.shape}")
    
    # Forward pass
    logits = model(dummy_input)
    
    print(f"Output Logits Shape: {logits.shape}")
    
    # Assertions
    assert logits.shape == (2, 8), f"Expected shape (2, 8), got {logits.shape}"
    
    # Check predict method
    probs = model.predict(dummy_input)
    print(f"Output Probs Shape: {probs.shape}")
    assert torch.allclose(probs.sum(dim=1), torch.ones(2)), "Probabilities must sum to 1"
    
    print("CRNN Forward Pass Test Passed Successfully!")

if __name__ == "__main__":
    test_crnn()
