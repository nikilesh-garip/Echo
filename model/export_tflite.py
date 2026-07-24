import os
import torch
from model import EchoCRNN

def export_onnx():
    print("Exporting Echo CRNN to ONNX format (Fixed Shape)...")
    
    model_path = "checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}. Please run train.py first.")
        
    # Load model
    device = torch.device("cpu")
    model = EchoCRNN(num_classes=8)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 5-second window yields T=156 frames at 16kHz
    dummy_input = torch.randn(1, 1, 64, 156)
    
    onnx_path = "checkpoints/echo_model.onnx"
    
    print("Running ONNX export...")
    # Export without dynamic axes to guarantee stable execution on edge runtimes
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14, # Highly compatible opset for mobile runtimes
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    
    print(f"ONNX Model successfully exported to: {onnx_path}")
    
    # Size comparison
    pth_size = os.path.getsize(model_path) / (1024 * 1024)
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"Original PyTorch Checkpoint Size: {pth_size:.4f} MB")
    print(f"Exported ONNX Model Size:        {onnx_size:.4f} MB")

if __name__ == "__main__":
    export_onnx()
