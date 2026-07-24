import os
import torch
from model import EchoCRNN

def export_openvino():
    print("Exporting Echo CRNN to OpenVINO Intermediate Representation (IR)...")
    
    model_path = "checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}. Please run train.py first.")
        
    try:
        import openvino as ov
    except ImportError:
        print("openvino package not found. Installing now...")
        os.system("pip install openvino")
        import openvino as ov
        
    # Load model
    device = torch.device("cpu")
    model = EchoCRNN(num_classes=8)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Trace model with dummy input
    dummy_input = torch.randn(1, 1, 64, 63)
    
    # Convert PyTorch model to OpenVINO Model representation
    ov_model = ov.convert_model(model, example_input=dummy_input)
    
    # Save openvino model files (.xml and .bin)
    output_dir = "checkpoints/openvino"
    os.makedirs(output_dir, exist_ok=True)
    ov.save_model(ov_model, os.path.join(output_dir, "echo_model.xml"))
    
    print(f"OpenVINO IR model successfully saved under: {output_dir}/")
    print(f"  - XML file: {output_dir}/echo_model.xml")
    print(f"  - BIN file: {output_dir}/echo_model.bin")

if __name__ == "__main__":
    export_openvino()
