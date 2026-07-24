import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import EchoDataset, get_dataloaders, PREPROCESSING_CONFIG
from model import EchoCRNN
import random

# For reproducibility
torch.manual_seed(42)
random.seed(42)

def load_metadata(metadata_path):
    data_list = []
    with open(metadata_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) == 2:
                file_path, label = parts
                data_list.append((file_path, label))
    return data_list

def train_model():
    print("Starting Echo Model Training...")
    
    metadata_path = "data/processed/metadata.csv"
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}. Please run ingest_real_datasets.py and generate_real_metadata.py first.")
        
    data_list = load_metadata(metadata_path)
    random.shuffle(data_list)
    
    # Split: 70% Train, 15% Val, 15% Test
    n_samples = len(data_list)
    n_train = int(0.70 * n_samples)
    n_val = int(0.15 * n_samples)
    
    train_data = data_list[:n_train]
    val_data = data_list[n_train:n_train+n_val]
    test_data = data_list[n_train+n_val:]
    
    print(f"Dataset Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Data loaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_data, val_data, test_data, batch_size=16
    )
    
    # Initialize Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = EchoCRNN(num_classes=8).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 15
    best_val_loss = float("inf")
    
    # Metric history log
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_loss / len(train_data)
        
        # Validation pass
        model.eval()
        running_val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
                
        epoch_val_loss = running_val_loss / len(val_data)
        epoch_val_acc = correct.double().item() / len(val_data)
        
        metrics["train_loss"].append(epoch_train_loss)
        metrics["val_loss"].append(epoch_val_loss)
        metrics["val_acc"].append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
        
        # Save best model checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("--> Saved best model checkpoint.")
            
    # Save training summary/metrics
    torch.save({
        "metrics": metrics,
        "config": PREPROCESSING_CONFIG,
        "test_data": test_data # Save test set split for evaluate.py
    }, "checkpoints/training_results.pth")
    
    print("Training Complete! Saved best_model.pth and training_results.pth")

if __name__ == "__main__":
    train_model()
