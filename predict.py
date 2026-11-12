"""
predict.py - Single-image CLI inference script.
Usage:
    python predict.py --image path/to/sample.png --checkpoint checkpoints/best_model.pt
"""
import argparse
from PIL import Image
import torch
from torchvision import transforms
from model import build_resnet18, CustomCNN

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

def predict_single_image(image_path: str, checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    arch = checkpoint.get("arch", "resnet18")

    model = build_resnet18(num_classes=2) if arch == "resnet18" else CustomCNN(num_classes=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Invert class_to_idx mapping: {0: 'defective', 1: 'normal'}
    idx_to_class = {v: k for k, v in checkpoint.get("class_to_idx", {"defective": 0, "normal": 1}).items()}

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probabilities).item()

    pred_label = idx_to_class[pred_idx]
    confidence = probabilities[pred_idx].item()

    print("\n" + "-"*40)
    print(f"Image:       {image_path}")
    print(f"Prediction:  {pred_label.upper()}")
    print(f"Confidence:  {confidence * 100:.2f}%")
    print("Probabilities:")
    for idx, class_name in idx_to_class.items():
        print(f"  • {class_name:10s}: {probabilities[idx].item()*100:.2f}%")
    print("-"*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionGuard Single-Image Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt", help="Path to .pt weights")
    args = parser.parse_args()

    predict_single_image(args.image, args.checkpoint)