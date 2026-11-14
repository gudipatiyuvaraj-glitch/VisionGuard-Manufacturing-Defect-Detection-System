"""
generate_sample_data.py
Industrial Synthetic Defect Dataset Generator for VisionGuard.
Generates realistic manufacturing anomalies: scratches, cracks, contamination, and structural defects.
"""
import os
import random
import argparse
from typing import Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def parse_args():
    parser = argparse.ArgumentParser(description="VisionGuard Synthetic Data Generator")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory for dataset")
    parser.add_argument("--train_samples", type=int, default=140, help="Total training samples (split evenly)")
    parser.add_argument("--val_samples", type=int, default=30, help="Total validation samples")
    parser.add_argument("--test_samples", type=int, default=30, help="Total test samples")
    parser.add_argument("--img_size", type=int, default=224, help="Image resolution (width/height)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def generate_base_component(size: int) -> Tuple[Image.Image, Tuple[int, int, int]]:
    """Creates a base industrial metallic component with realistic texture and noise."""
    base_val = random.randint(185, 215)
    img_array = np.full((size, size, 3), base_val, dtype=np.uint8)
    noise = np.random.normal(0, 7, (size, size, 3)).astype(np.int16)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    
    center = size // 2
    outer_r = int(size * 0.38)
    inner_r = int(size * 0.18)
    
    draw.ellipse([center - outer_r, center - outer_r, center + outer_r, center + outer_r],
                 outline=(70, 70, 70), width=4)
    draw.ellipse([center - inner_r, center - inner_r, center + inner_r, center + inner_r],
                 fill=(110, 110, 110), outline=(50, 50, 50), width=3)
                 
    return img, (center, outer_r, inner_r)

def inject_defect(img: Image.Image, center_info: Tuple[int, int, int]) -> Image.Image:
    """Injects one of four distinct industrial defect classes."""
    draw = ImageDraw.Draw(img)
    center, outer_r, inner_r = center_info
    defect_type = random.choice(["scratch", "crack", "contamination", "dent"])

    if defect_type == "scratch":
        x1 = random.randint(center - outer_r + 15, center + outer_r - 15)
        y1 = random.randint(center - outer_r + 15, center + outer_r - 15)
        length = random.randint(30, 70)
        angle = random.uniform(0, 2 * np.pi)
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        draw.line([x1, y1, x2, y2], fill=(25, 25, 25), width=random.randint(2, 4))

    elif defect_type == "crack":
        curr_x = random.randint(center - outer_r + 25, center + outer_r - 25)
        curr_y = random.randint(center - outer_r + 25, center + outer_r - 25)
        points = [(curr_x, curr_y)]
        for _ in range(random.randint(4, 7)):
            curr_x += random.randint(-16, 16)
            curr_y += random.randint(-16, 16)
            points.append((curr_x, curr_y))
        draw.line(points, fill=(15, 15, 15), width=2)

    elif defect_type == "contamination":
        spot_x = random.randint(center - outer_r + 20, center + outer_r - 20)
        spot_y = random.randint(center - outer_r + 20, center + outer_r - 20)
        rad = random.randint(8, 18)
        draw.ellipse([spot_x - rad, spot_y - rad, spot_x + rad, spot_y + rad], fill=(45, 30, 20))
        img = img.filter(ImageFilter.GaussianBlur(radius=0.7))

    elif defect_type == "dent":
        dent_x = center + int(outer_r * np.cos(random.uniform(0, 2 * np.pi)))
        dent_y = center + int(outer_r * np.sin(random.uniform(0, 2 * np.pi)))
        dr = random.randint(10, 16)
        draw.ellipse([dent_x - dr, dent_y - dr, dent_x + dr, dent_y + dr], fill=(80, 80, 80), outline=(30, 30, 30))

    return img

def build_dataset(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    splits = {
        "train": (args.train_samples // 2, args.train_samples // 2),
        "val":   (args.val_samples // 2, args.val_samples // 2),
        "test":  (args.test_samples // 2, args.test_samples // 2)
    }

    print(f"[*] Initializing synthetic dataset generation at: {args.output_dir}")
    for split_name, (norm_cnt, def_cnt) in splits.items():
        for category, count in [("normal", norm_cnt), ("defective", def_cnt)]:
            dir_path = os.path.join(args.output_dir, split_name, category)
            os.makedirs(dir_path, exist_ok=True)
            
            for i in range(1, count + 1):
                img, center_info = generate_base_component(args.img_size)
                if category == "defective":
                    img = inject_defect(img, center_info)
                
                filename = f"{category}_{i:04d}.png"
                img.save(os.path.join(dir_path, filename))

            print(f"  ✓ {split_name.capitalize():5s} | {category:9s} : {count} images written.")

    print("[*] Dataset generation completed successfully.\n")

if __name__ == "__main__":
    args = parse_args()
    build_dataset(args)