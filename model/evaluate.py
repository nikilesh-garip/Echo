import os
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from dataset import EchoDataset, PREPROCESSING_CONFIG
from model import EchoTransformer
from torch.utils.data import DataLoader

def evaluate_model():
    print("Starting Echo Model Evaluation...")
    
    results_path = "checkpoints/training_results.pth"
    model_path = "checkpoints/best_model.pth"
    
    if not os.path.exists(results_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Training outputs not found. Please run train.py first.")
        
    # Load training details (including test set split)
    checkpoint = torch.load(results_path)
    test_data = checkpoint["test_data"]
    config = checkpoint.get("config", PREPROCESSING_CONFIG)
    
    # Initialize dataset and loader
    test_dataset = EchoDataset(test_data, config=config, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EchoTransformer(num_classes=8).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Standard metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(8)))
    
    # Compute FPR (False Positive Rate) & FNR (False Negative Rate) per class
    # FPR = FP / (FP + TN)
    # FNR = FN / (FN + TP)
    fprs = []
    fnrs = []
    for i in range(8):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fn - fp
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fprs.append(fpr)
        fnrs.append(fnr)
        
    class_names = ["Normal", "Gunshot", "Explosion", "Scream", "Glass Breaking", "Fire Alarm", "Siren", "Shouting"]
    
    print("\n================ EVALUATION METRICS ================")
    print(f"Overall Accuracy:  {acc:.4f}")
    print(f"Macro Precision:   {precision:.4f}")
    print(f"Macro Recall:      {recall:.4f}")
    print(f"Macro F1 Score:    {f1:.4f}")
    print("====================================================")
    
    print("\nClass-wise Metrics:")
    print(f"{'Class':<20} | {'FPR':<8} | {'FNR':<8}")
    print("-" * 42)
    for i, name in enumerate(class_names):
        print(f"{name:<20} | {fprs[i]:.4f}   | {fnrs[i]:.4f}")
        
    print("\nConfusion Matrix:")
    print("   " + "   ".join(f"C{i}" for i in range(8)))
    for i, row in enumerate(cm):
        print(f"C{i} " + " ".join(f"{val:4d}" for val in row) + f"  ({class_names[i]})")
        
    # Write report file
    os.makedirs("../reports", exist_ok=True)
    report_path = "../reports/evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("Echo CRNN Model Evaluation Report\n")
        f.write("=================================\n\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write("Class-wise Metrics:\n")
        for i, name in enumerate(class_names):
            f.write(f"{name}: FPR={fprs[i]:.4f}, FNR={fnrs[i]:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm) + "\n")
        
    print(f"\nEvaluation Report successfully saved to: reports/evaluation_report.txt")

if __name__ == "__main__":
    evaluate_model()
