"""
generate_sample_data.py
Procedurally generates synthetic industrial manufacturing images:
- Normal: Uniform metallic disc with clean surface
- Defective: Metallic disc with cracks, scratches, or holes
"""
import os
import random
import numpy as np
from PIL import Image, ImageDraw

def create_synthetic_image(is_defect: bool, size: int = 224) -> Image.Image:
    # 1. Base metallic background with noise
    base_color = random.randint(180, 210)
    img_array = np.full((size, size, 3), base_color, dtype=np.uint8)
    noise = np.random.normal(0, 8, (size, size, 3)).astype(np.int16)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    
    # 2. Draw circular industrial product (e.g., metal washer/bearing)
    center = size // 2
    radius = size // 3
    draw.ellipse(
        [center - radius, center - radius, center + radius, center + radius],
        outline=(80, 80, 80),
        width=4
    )
    inner_radius = radius // 2
    draw.ellipse(
        [center - inner_radius, center - inner_radius, center + inner_radius, center + inner_radius],
        fill=(120, 120, 120),
        outline=(60, 60, 60),
        width=3
    )

    # 3. Add defects if defective
    if is_defect:
        defect_type = random.choice(["scratch", "crack", "burn_hole"])
        
        if defect_type == "scratch":
            # Dark scratch lines
            x1, y1 = random.randint(50, 170), random.randint(50, 170)
            x2, y2 = x1 + random.randint(-40, 40), y1 + random.randint(-40, 40)
            draw.line([x1, y1, x2, y2], fill=(20, 20, 20), width=3)
            
        elif defect_type == "crack":
            # Jagged crack
            points = [(random.randint(70, 150), random.randint(70, 150))]
            for _ in range(4):
                last_pt = points[-1]
                points.append((last_pt[0] + random.randint(-15, 15), last_pt[1] + random.randint(-15, 15)))
            draw.line(points, fill=(10, 10, 10), width=2)
            
        elif defect_type == "burn_hole":
            # Dark oxidized spot
            bx, by = random.randint(80, 140), random.randint(80, 140)
            br = random.randint(6, 14)
            draw.ellipse([bx - br, by - br, bx + br, by + br], fill=(30, 15, 15))

    return img

def generate_dataset(base_dir: str = "./data"):
    # Split counts: 70% Train (140), 15% Val (30), 15% Test (30)
    splits = {
        "train": {"normal": 70, "defective": 70},
        "val":   {"normal": 15, "defective": 15},
        "test":  {"normal": 15, "defective": 15}
    }

    print("Generating synthetic manufacturing dataset...")
    for split, counts in splits.items():
        for category, count in counts.items():
            folder_path = os.path.join(base_dir, split, category)
            os.makedirs(folder_path, exist_ok=True)
            
            is_defect = (category == "defective")
            for i in range(1, count + 1):
                img = create_synthetic_image(is_defect=is_defect)
                filename = f"{category}_{i:03d}.png"
                img.save(os.path.join(folder_path, filename))
                
            print(f"  ✓ Created {count} images in {split}/{category}/")

    print("\nDataset ready inside ./data/")

if __name__ == "__main__":
    generate_dataset()