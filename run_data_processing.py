import os
import sys
from pathlib import Path
from data_processor import PDestreDataProcessor

def main():
    """Run data processing with default parameters"""
    
    # Default parameters
    dataset_path = "../P-DESTRE"
    output_path = "../processed_pdestre"
    frame_rate = 10  # Extract every 10th frame to reduce data size
    min_confidence = 0.5
    min_box_size = 20
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    seed = 42
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' not found!")
        print("Please make sure the P-DESTRE dataset is in the current directory.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    print("="*60)
    print("P-DESTRE Dataset Processing for YOLO Training")
    print("="*60)
    print(f"Dataset path: {dataset_path}")
    print(f"Output path: {output_path}")
    print(f"Frame rate: {frame_rate} (extract every {frame_rate}th frame)")
    print(f"Min confidence: {min_confidence}")
    print(f"Min box size: {min_box_size}px")
    print(f"Train/Val/Test split: {train_ratio}/{val_ratio}/{test_ratio}")
    print("="*60)
    
    try:
        # Create processor
        processor = PDestreDataProcessor(
            dataset_path=dataset_path,
            output_path=output_path,
            frame_rate=frame_rate,
            min_confidence=min_confidence,
            min_box_size=min_box_size
        )
        
        # Process dataset
        processor.process_dataset(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed
        )
        
        print("\n Data processing completed successfully!")
        print(f" Processed data saved to: {output_path}")
        print(f" YOLO config file: {output_path}/config/dataset.yaml")
        print(f" Dataset info: {output_path}/dataset_info.json")
        
    except Exception as e:
        print(f"\n Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 