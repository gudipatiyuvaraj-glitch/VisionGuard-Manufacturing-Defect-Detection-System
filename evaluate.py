"""
evaluate.py - Test evaluation, Confusion Matrix, and ROC-AUC curve generation.
"""
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from dataset import get_dataloaders
from model import build_resnet18, CustomCNN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_resnet18() if ckpt.get("arch") == "resnet18" else CustomCNN()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    _, _, test_loader, _ = get_dataloaders(args.data_dir, batch_size=32)

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    # Metrics
    print("\n" + "="*50)
    print("DETAILED TEST CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=["Defective", "Normal"], digits=4))

    auc = roc_auc_score(all_labels, all_probs)
    print(f"ROC-AUC Score: {auc:.4f}\n")

    # 1. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Defective", "Normal"], yticklabels=["Defective", "Normal"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "roc_curve.png"), dpi=300)
    plt.close()
    print(f"[✓] Visualizations saved to {args.output_dir}/")

if __name__ == "__main__":
    main()