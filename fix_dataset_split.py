#!/usr/bin/env python3
"""
Fix Dataset Split - Ensure each split has data
This script redistributes data to ensure train/val/test splits all have images
"""

import os
import shutil
import random
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_dataset_split(dataset_path: str, train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1):
    """
    Fix dataset split by redistributing images to ensure each split has data
    
    Args:
        dataset_path: Path to the processed dataset
        train_ratio: Ratio for training data
        val_ratio: Ratio for validation data  
        test_ratio: Ratio for testing data
    """
    dataset_path = Path(dataset_path)
    
    # Check if dataset exists
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return
    
    # Get all images from train directory (since it has data)
    train_images_dir = dataset_path / 'images' / 'train'
    train_labels_dir = dataset_path / 'labels' / 'train'
    
    if not train_images_dir.exists():
        logger.error(f"Train images directory does not exist: {train_images_dir}")
        return
    
    # Get all image files
    image_files = list(train_images_dir.glob('*.jpg'))
    logger.info(f"Found {len(image_files)} images in train directory")
    
    if len(image_files) == 0:
        logger.error("No images found in train directory")
        return
    
    # Shuffle images
    random.seed(42)
    random.shuffle(image_files)
    
    # Calculate split sizes
    total_images = len(image_files)
    n_train = int(total_images * train_ratio)
    n_val = int(total_images * val_ratio)
    n_test = total_images - n_train - n_val  # Remaining for test
    
    logger.info(f"Redistributing {total_images} images:")
    logger.info(f"  Train: {n_train} images")
    logger.info(f"  Val: {n_val} images") 
    logger.info(f"  Test: {n_test} images")
    
    # Split images
    train_images = image_files[:n_train]
    val_images = image_files[n_train:n_train + n_val]
    test_images = image_files[n_train + n_val:]
    
    # Create directories if they don't exist
    val_images_dir = dataset_path / 'images' / 'val'
    test_images_dir = dataset_path / 'images' / 'test'
    val_labels_dir = dataset_path / 'labels' / 'val'
    test_labels_dir = dataset_path / 'labels' / 'test'
    
    val_images_dir.mkdir(parents=True, exist_ok=True)
    test_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)
    test_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Move images and labels to validation
    logger.info("Moving images to validation split...")
    for img_path in val_images:
        # Move image
        new_img_path = val_images_dir / img_path.name
        shutil.move(str(img_path), str(new_img_path))
        
        # Move corresponding label
        label_path = train_labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            new_label_path = val_labels_dir / f"{img_path.stem}.txt"
            shutil.move(str(label_path), str(new_label_path))
    
    # Move images and labels to test
    logger.info("Moving images to test split...")
    for img_path in test_images:
        # Move image
        new_img_path = test_images_dir / img_path.name
        shutil.move(str(img_path), str(new_img_path))
        
        # Move corresponding label
        label_path = train_labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            new_label_path = test_labels_dir / f"{img_path.stem}.txt"
            shutil.move(str(label_path), str(new_label_path))
    
    # Count final images in each split
    final_train_images = len(list(train_images_dir.glob('*.jpg')))
    final_val_images = len(list(val_images_dir.glob('*.jpg')))
    final_test_images = len(list(test_images_dir.glob('*.jpg')))
    
    logger.info("="*50)
    logger.info("DATASET SPLIT FIXED SUCCESSFULLY")
    logger.info("="*50)
    logger.info(f"Train images: {final_train_images}")
    logger.info(f"Val images: {final_val_images}")
    logger.info(f"Test images: {final_test_images}")
    logger.info(f"Total images: {final_train_images + final_val_images + final_test_images}")
    logger.info("="*50)
    
    # Update dataset.yaml to use absolute paths
    update_dataset_yaml(dataset_path)
    
    logger.info("✅ Dataset split fixed! You can now train your YOLO model.")

def update_dataset_yaml(dataset_path: Path):
    """Update dataset.yaml with absolute paths"""
    yaml_path = dataset_path / 'config' / 'dataset.yaml'
    
    if yaml_path.exists():
        # Read current yaml
        with open(yaml_path, 'r') as f:
            content = f.read()
        
        # Replace relative paths with absolute paths
        content = content.replace('path: processed_data', f'path: {dataset_path.absolute()}')
        content = content.replace('path: limited_processed_data', f'path: {dataset_path.absolute()}')
        
        # Write updated yaml
        with open(yaml_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Updated dataset.yaml with absolute paths")

def main():
    """Main function"""
    dataset_path = "limited_processed_data"
    
    print("="*60)
    print("FIXING DATASET SPLIT")
    print("="*60)
    print(f"Dataset path: {dataset_path}")
    print("This will redistribute images to ensure each split has data")
    print("="*60)
    
    try:
        fix_dataset_split(dataset_path)
        print("\n✅ Dataset split fixed successfully!")
        print("You can now train your YOLO model with:")
        print(f"yolo train data={dataset_path}/config/dataset.yaml model=yolov8n.pt epochs=5")
        
    except Exception as e:
        print(f"\n❌ Error fixing dataset split: {e}")

if __name__ == "__main__":
    main() 