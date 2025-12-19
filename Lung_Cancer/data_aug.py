# Applying Data augmentation Techniques for Lung Cancer Imaging Data

import os
import random 
from PIL import Image, ImageEnhance, ImageOps

# ------Configuration------
folder_path = 'F:\\Machine Learning\\PyTorch\\Lung_Cancer\\data\\Normal'
target_count = '2131' # Target number of images after augmentation

# ------Data Augmentation Functions------

def augment_folder(folder, target):
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_count = len(files)
    augmentations_needed = target - current_count

    print(f"Current images: {current_count}")
    print(f"Augmentations needed: {augmentations_needed}")
    print(f"Generating {augmentations_needed} new images...")

    if augmentations_needed <= 0:
        print("No augmentation needed.")
        return
    
    # loop until we reach the target count
    generated = 0
    while generated < augmentations_needed:

        # pick a random image to clone
        random_file = random.choice(files)
        img_path = os.path.join(folder, random_file)

        try:
            with Image.open(img_path) as img:

                # Apply random augmentation
                # 1. Random Rotation (-15 to 15 degrees)
                angle = random.uniform(-15, 15)
                new_img = img.rotate(angle)

                # 2. Random Flip (left, right)
                if random.choice([True, False]):
                    new_img = ImageOps.mirror(new_img)

                # 3. Random Brightness Adjustment (0.7 to 1.3)
                enhancer = ImageEnhance.Brightness(new_img)
                new_img = enhancer.enhance(random.uniform(0.7, 1.3))

                # Save the new image
                save_name = f"aug_{generated}_{random_file}"
                new_img.save(os.path.join(folder, save_name))
                generated += 1

                if generated % 100 == 0:
                    print(f"Creted {generated} images...")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"Augmentation complete. Total images now: {len(os.listdir(folder))}")

# ------Execute Augmentation------
augment_folder(folder_path, int(target_count))