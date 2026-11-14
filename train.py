"""
train.py - VisionGuard Training Pipeline
Features: ResNet-18 / Custom CNN, Resumable Checkpoints (--resume), 
LR Scheduling, Early Stopping, TensorBoard Logging, and Training Curves.
"""
import os
import sys
import argparse
import traceback
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloaders
from model import build_resnet18, CustomCNN

def parse_args():
    parser = argparse.ArgumentParser(description="VisionGuard Model Training Script")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--arch", type=str, choices=["resnet18", "custom"], default="resnet18", help="Architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume training from")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save training curves")
    return parser.parse_args()

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Training Batch", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Validation Batch", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def save_training_curves(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(epochs, history["train_loss"], 'b-o', label='Train Loss')
    ax1.plot(epochs, history["val_loss"], 'r-o', label='Val Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["train_acc"], 'b-o', label='Train Acc')
    ax2.plot(epochs, history["val_acc"], 'r-o', label='Val Acc')
    ax2.set_title('Training & Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    curve_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"[*] Training curves exported to: {curve_path}", flush=True)

def main():
    args = parse_args()
    print("[*] Starting VisionGuard training script...", flush=True)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute device: {device}", flush=True)

    # Use num_workers=0 to prevent Windows silent crashes
    train_loader, val_loader, _, class_to_idx = get_dataloaders(
        args.data_dir, 
        batch_size=args.batch_size, 
        num_workers=0
    )
    print(f"[*] Loaded dataset classes: {class_to_idx}", flush=True)

    # Architecture
    if args.arch == "resnet18":
        model = build_resnet18(num_classes=2)
    else:
        model = CustomCNN(num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    writer = SummaryWriter(log_dir="./runs/visionguard_experiment")

    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # RESUME CAPABILITY
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"[*] Loading checkpoint from: {args.resume}", flush=True)
            ckpt = torch.load(args.resume, map_location=device)
            start_epoch = ckpt["epoch"] + 1
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            best_val_loss = ckpt.get("val_loss", float("inf"))
            print(f"[✓] Checkpoint restored. Resuming training from epoch {start_epoch}", flush=True)
        else:
            print(f"[!] Warning: Checkpoint path '{args.resume}' not found. Starting from epoch 1.", flush=True)

    print(f"\n--- Training {args.arch.upper()} for {args.epochs} Epochs ---", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate(model, val_loader, criterion, device)
        scheduler.step(v_loss)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        writer.add_scalars("Loss", {"Train": t_loss, "Validation": v_loss}, epoch)
        writer.add_scalars("Accuracy", {"Train": t_acc, "Validation": v_acc}, epoch)

        print(f"Epoch [{epoch:02d}/{args.epochs}] | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}", flush=True)

        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": v_loss,
            "class_to_idx": class_to_idx,
            "arch": args.arch
        }
        
        # Save last checkpoint for resume capability
        torch.save(checkpoint_state, os.path.join(args.checkpoint_dir, "last_checkpoint.pt"))

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
            torch.save(checkpoint_state, os.path.join(args.checkpoint_dir, "best_model.pt"))
            print(f"  [*] Best model saved at epoch {epoch}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[!] Early stopping triggered at epoch {epoch}", flush=True)
                break

    writer.close()
    save_training_curves(history, args.output_dir)
    print("\n[✓] Training complete.", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[!] Fatal Error: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)