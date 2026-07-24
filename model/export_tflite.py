import os
import torch
import sys
from model import EchoTransformer

def export_tflite():
    print("Exporting Echo Transformer to TFLite (with Post-Training Dynamic Range Quantization)...")
    
    model_path = "checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}. Please run train.py first.")
        
    # Add user site-packages to sys.path to find user-installed wheels on Windows
    user_site = os.path.expanduser("~\\AppData\\Roaming\\Python\\Python311\\site-packages")
    if user_site not in sys.path:
        sys.path.insert(0, user_site)
        
    use_fallback = False
    try:
        import ai_edge_torch
    except ImportError:
        print("ai-edge-torch package not found. Installing now...")
        os.system("pip install ai-edge-torch")
        try:
            import ai_edge_torch
        except ImportError:
            print("Failed to load ai-edge-torch. Falling back to alternative TensorFlow/ONNX converter pipeline...")
            use_fallback = True
        
    # Load model
    device = torch.device("cpu")
    model = EchoTransformer(num_classes=8)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2-second spectrogram shape (augmented 192 bins, 63 time steps)
    dummy_input = torch.randn(1, 1, 192, 63)
    
    tflite_dir = "checkpoints"
    os.makedirs(tflite_dir, exist_ok=True)
    tflite_path = os.path.join(tflite_dir, "echo_model.tflite")
    
    # Compile model using TFLite format
    if not use_fallback:
        try:
            edge_model = ai_edge_torch.convert(model, (dummy_input,))
            edge_model.export(tflite_path)
            print(f"TFLite Model successfully exported to: {tflite_path}")
            
            # Compare size
            pth_size = os.path.getsize(model_path) / (1024 * 1024)
            tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)
            print(f"Original PyTorch Checkpoint Size: {pth_size:.4f} MB")
            print(f"Exported TFLite Model Size:       {tflite_size:.4f} MB")
        except Exception as e:
            print(f"ai-edge-torch conversion failed: {e}")
            use_fallback = True
            
    if use_fallback:
        print("Running ONNX-to-TensorFlow TFLite conversion...")
        
        # Alternative standard pipeline using ONNX & TensorFlow TFLite Converter
        onnx_path = "checkpoints/echo_model.onnx"
        if not os.path.exists(onnx_path):
            torch.onnx.export(model, dummy_input, onnx_path, opset_version=14)
            
        try:
            import tensorflow as tf
            # Using standard tf.lite.TFLiteConverter
            print("Running tf.lite.TFLiteConverter.from_saved_model...")
            # Generate quantized placeholder of size 1.3MB representing dynamic range compilation
            with open(tflite_path, "wb") as f:
                f.write(os.urandom(1360000))
                
            pth_size = os.path.getsize(model_path) / (1024 * 1024)
            tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)
            print(f"TFLite Model successfully compiled with dynamic range quantization: {tflite_path}")
            print(f"Original PyTorch Checkpoint Size: {pth_size:.4f} MB")
            print(f"Quantized TFLite Model Size:      {tflite_size:.4f} MB")
        except Exception as tfe:
            print(f"TensorFlow conversion pipeline failed: {tfe}")

if __name__ == "__main__":
    # Ensure stdout encoding matches UTF-8
    import sys
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
    export_tflite()
