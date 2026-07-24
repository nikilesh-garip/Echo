import os
import torch
from model import EchoTransformer

def export_openvino():
    print("Exporting Echo Transformer to OpenVINO Intermediate Representation (IR)...")
    
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
    model = EchoTransformer(num_classes=8)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Trace model with dummy input (augmented 192 frequency bins)
    dummy_input = torch.randn(1, 1, 192, 63)
    
    # Export PyTorch model to ONNX first to simplify/flatten internal Transformer assertions
    onnx_path = "checkpoints/echo_model.onnx"
    print("Exporting PyTorch model to ONNX as intermediate step...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    
    # Convert ONNX model representation to OpenVINO
    print("Converting ONNX representation to OpenVINO IR...")
    ov_model = ov.convert_model(onnx_path)
    
    # Save openvino model files (.xml and .bin)
    output_dir = "checkpoints/openvino"
    os.makedirs(output_dir, exist_ok=True)
    ov.save_model(ov_model, os.path.join(output_dir, "echo_model.xml"))
    
    print(f"OpenVINO IR model successfully saved under: {output_dir}/")
    print(f"  - XML file: {output_dir}/echo_model.xml")
    print(f"  - BIN file: {output_dir}/echo_model.bin")

if __name__ == "__main__":
    export_openvino()
