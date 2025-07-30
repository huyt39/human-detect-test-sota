#!/usr/bin/env python3
"""
Limited Video Extraction for P-DESTRE Dataset
Extract frames from only 15 random videos to reduce dataset size
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import random
from tqdm import tqdm
import logging
import yaml
import json
from typing import List, Tuple, Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LimitedVideoExtractor:
    """
    Extract frames from limited number of videos to reduce dataset size
    """
    
    def __init__(self, dataset_path: str, output_path: str, num_videos: int = 15,
                 frame_rate: int = 5, min_confidence: float = 0.5, min_box_size: int = 20):
        """
        Initialize the limited video extractor
        
        Args:
            dataset_path: Path to P-DESTRE dataset
            output_path: Path to save processed data
            num_videos: Number of videos to process (default: 15)
            frame_rate: Frame extraction rate (1 = every frame, 5 = every 5th frame, etc.)
            min_confidence: Minimum confidence threshold for detections
            min_box_size: Minimum bounding box size (width or height) in pixels
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.num_videos = num_videos
        self.frame_rate = frame_rate
        self.min_confidence = min_confidence
        self.min_box_size = min_box_size
        
        # Create output directories
        self.create_output_directories()
        
        # YOLO class mapping (0 = person)
        self.class_mapping = {'person': 0}
        
        # Video extensions
        self.video_extensions = ['.MP4', '.mp4', '.avi', '.mov']
        
        # P-DESTRE annotation columns
        self.annotation_columns = [
            'frame_id', 'track_id', 'x', 'y', 'width', 'height', 'confidence',
            'x_3d', 'y_3d', 'z_3d', 'occlusion', 'truncation', 'orientation',
            'motion_blur', 'illumination', 'weather', 'scene', 'time_of_day',
            'age', 'gender', 'clothing', 'action', 'interaction', 'group_size',
            'group_type', 'group_activity', 'group_density', 'group_cohesion'
        ]
        
    def create_output_directories(self):
        """Create necessary output directories"""
        dirs = [
            self.output_path / 'images' / 'train',
            self.output_path / 'images' / 'val', 
            self.output_path / 'images' / 'test',
            self.output_path / 'labels' / 'train',
            self.output_path / 'labels' / 'val',
            self.output_path / 'labels' / 'test',
            self.output_path / 'config'
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def get_video_files(self) -> List[Path]:
        """Get all video files from the dataset"""
        video_files = []
        videos_dir = self.dataset_path / 'videos'
        
        if not videos_dir.exists():
            raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
        
        for ext in self.video_extensions:
            video_files.extend(videos_dir.glob(f"*{ext}"))
        
        logger.info(f"Found {len(video_files)} total video files")
        return video_files
    
    def select_random_videos(self, video_files: List[Path], seed: int = 42) -> List[Path]:
        """Select random videos from the dataset"""
        random.seed(seed)
        
        if len(video_files) <= self.num_videos:
            logger.warning(f"Only {len(video_files)} videos found, using all of them")
            selected_videos = video_files
        else:
            selected_videos = random.sample(video_files, self.num_videos)
        
        logger.info(f"Selected {len(selected_videos)} videos for processing")
        for i, video in enumerate(selected_videos, 1):
            logger.info(f"  {i}. {video.name}")
        
        return selected_videos
    
    def parse_annotation_file(self, annotation_file: Path) -> pd.DataFrame:
        """
        Parse P-DESTRE annotation file and return DataFrame
        
        Args:
            annotation_file: Path to annotation file
            
        Returns:
            DataFrame with filtered annotations
        """
        try:
            # Read annotation file without header
            df = pd.read_csv(annotation_file, header=None)
            
            # Assign column names
            df.columns = self.annotation_columns[:len(df.columns)]
            
            # Filter valid detections
            valid_mask = (
                (df['confidence'] >= self.min_confidence) &  # Confidence threshold
                (df['width'] >= self.min_box_size) &         # Minimum width
                (df['height'] >= self.min_box_size) &        # Minimum height
                (df['track_id'] != -1) &                     # Valid track ID
                (df['x'] >= 0) & (df['y'] >= 0) &            # Valid coordinates
                (df['width'] > 0) & (df['height'] > 0)       # Valid dimensions
            )
            
            df_filtered = df[valid_mask].copy()
            
            logger.info(f"Filtered {len(df)} -> {len(df_filtered)} valid annotations from {annotation_file.name}")
            
            return df_filtered
            
        except Exception as e:
            logger.error(f"Error parsing annotation file {annotation_file}: {e}")
            return pd.DataFrame()
    
    def convert_to_yolo_format(self, x: float, y: float, width: float, height: float, 
                              img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        Convert bounding box coordinates to YOLO format (normalized)
        
        Args:
            x, y: Top-left corner coordinates
            width, height: Bounding box dimensions
            img_width, img_height: Image dimensions
            
        Returns:
            Tuple of (x_center, y_center, width, height) in normalized coordinates
        """
        # Convert to center coordinates
        x_center = x + width / 2
        y_center = y + height / 2
        
        # Normalize coordinates
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = width / img_width
        height_norm = height / img_height
        
        # Ensure coordinates are within [0, 1]
        x_center_norm = max(0, min(1, x_center_norm))
        y_center_norm = max(0, min(1, y_center_norm))
        width_norm = max(0, min(1, width_norm))
        height_norm = max(0, min(1, height_norm))
        
        return x_center_norm, y_center_norm, width_norm, height_norm
    
    def extract_frames_and_annotations(self, video_file: Path, annotation_file: Path, 
                                     split: str) -> int:
        """
        Extract frames from video and corresponding annotations
        
        Args:
            video_file: Path to video file
            annotation_file: Path to annotation file
            split: Data split ('train', 'val', 'test')
            
        Returns:
            Number of frames extracted
        """
        # Parse annotations
        annotations_df = self.parse_annotation_file(annotation_file)
        if annotations_df.empty:
            logger.warning(f"No valid annotations found in {annotation_file}")
            return 0
        
        # Open video
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_file}")
            return 0
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Processing video: {video_file.name}")
        logger.info(f"FPS: {fps:.2f}, Total frames: {total_frames}, Resolution: {width}x{height}")
        
        frame_count = 0
        extracted_count = 0
        
        # Create progress bar
        pbar = tqdm(total=total_frames, desc=f"Extracting frames from {video_file.name}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame based on frame rate
            if frame_count % self.frame_rate == 0:
                # Get annotations for this frame (P-DESTRE uses 1-based frame indexing)
                frame_annotations = annotations_df[annotations_df['frame_id'] == frame_count + 1]
                
                if not frame_annotations.empty:
                    # Generate unique filename
                    video_name = video_file.stem
                    frame_filename = f"{video_name}_frame_{frame_count:06d}.jpg"
                    label_filename = f"{video_name}_frame_{frame_count:06d}.txt"
                    
                    # Save frame
                    frame_path = self.output_path / 'images' / split / frame_filename
                    cv2.imwrite(str(frame_path), frame)
                    
                    # Create YOLO annotation file
                    label_path = self.output_path / 'labels' / split / label_filename
                    self.create_yolo_annotation(frame_annotations, label_path, width, height)
                    
                    extracted_count += 1
            
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        logger.info(f"Extracted {extracted_count} frames from {video_file.name}")
        return extracted_count
    
    def create_yolo_annotation(self, frame_annotations: pd.DataFrame, 
                              label_path: Path, img_width: int, img_height: int):
        """
        Create YOLO format annotation file
        
        Args:
            frame_annotations: DataFrame with annotations for current frame
            label_path: Path to save YOLO annotation file
            img_width, img_height: Image dimensions
        """
        with open(label_path, 'w') as f:
            for _, annotation in frame_annotations.iterrows():
                # Convert to YOLO format
                x_center, y_center, width, height = self.convert_to_yolo_format(
                    annotation['x'], annotation['y'], 
                    annotation['width'], annotation['height'],
                    img_width, img_height
                )
                
                # Write YOLO format: class_id x_center y_center width height
                class_id = self.class_mapping['person']  # 0 for person
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def process_limited_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.2, 
                              test_ratio: float = 0.1, seed: int = 42):
        """
        Process limited number of videos
        
        Args:
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            seed: Random seed for reproducibility
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        # Set random seed
        random.seed(seed)
        
        # Get all video files
        all_video_files = self.get_video_files()
        
        if not all_video_files:
            raise ValueError("No video files found in the dataset")
        
        # Select random videos
        selected_videos = self.select_random_videos(all_video_files, seed)
        
        # Shuffle selected videos
        random.shuffle(selected_videos)
        
        # Split videos into train/val/test
        n_videos = len(selected_videos)
        n_train = int(n_videos * train_ratio)
        n_val = int(n_videos * val_ratio)
        
        train_videos = selected_videos[:n_train]
        val_videos = selected_videos[n_train:n_train + n_val]
        test_videos = selected_videos[n_train + n_val:]
        
        logger.info(f"Dataset split: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
        
        # Process each split
        splits = [
            ('train', train_videos),
            ('val', val_videos),
            ('test', test_videos)
        ]
        
        total_frames = 0
        split_stats = {}
        
        for split_name, videos in splits:
            logger.info(f"\nProcessing {split_name} split...")
            split_frames = 0
            processed_videos = 0
            
            for video_file in videos:
                # Find corresponding annotation file
                annotation_file = self.dataset_path / 'annotation' / f"{video_file.stem}.txt"
                
                if not annotation_file.exists():
                    logger.warning(f"Annotation file not found: {annotation_file}")
                    continue
                
                # Extract frames and annotations
                frames_extracted = self.extract_frames_and_annotations(
                    video_file, annotation_file, split_name
                )
                split_frames += frames_extracted
                if frames_extracted > 0:
                    processed_videos += 1
            
            split_stats[split_name] = {
                'videos': len(videos),
                'processed_videos': processed_videos,
                'frames': split_frames
            }
            logger.info(f"{split_name.capitalize()} split: {split_frames} frames extracted from {processed_videos}/{len(videos)} videos")
            total_frames += split_frames
        
        logger.info(f"\nTotal frames extracted: {total_frames}")
        
        # Create YOLO configuration file
        self.create_yolo_config()
        
        # Create dataset info file
        self.create_dataset_info(split_stats, total_frames, selected_videos)
        
        # Print summary
        self.print_summary(split_stats, total_frames, selected_videos)
    
    def create_yolo_config(self):
        """Create YOLO configuration file"""
        config = {
            "path": str(self.output_path),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(self.class_mapping),  # number of classes
            "names": list(self.class_mapping.keys())
        }
        
        config_path = self.output_path / 'config' / 'dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Created YOLO config file: {config_path}")
    
    def create_dataset_info(self, split_stats: Dict, total_frames: int, selected_videos: List[Path]):
        """Create dataset information file"""
        info = {
            "dataset_name": "P-DESTRE (Limited)",
            "task": "Human Detection",
            "format": "YOLO",
            "total_frames": total_frames,
            "num_videos_processed": len(selected_videos),
            "frame_rate": self.frame_rate,
            "min_confidence": self.min_confidence,
            "min_box_size": self.min_box_size,
            "splits": split_stats,
            "classes": self.class_mapping,
            "annotation_format": "YOLO (normalized coordinates)",
            "image_format": "JPG",
            "created_date": pd.Timestamp.now().isoformat(),
            "processing_parameters": {
                "num_videos": self.num_videos,
                "frame_rate": self.frame_rate,
                "min_confidence": self.min_confidence,
                "min_box_size": self.min_box_size
            },
            "selected_videos": [video.name for video in selected_videos]
        }
        
        info_path = self.output_path / 'dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Created dataset info file: {info_path}")
    
    def print_summary(self, split_stats: Dict, total_frames: int, selected_videos: List[Path]):
        """Print processing summary"""
        logger.info("\n" + "="*60)
        logger.info("LIMITED DATASET PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total videos processed: {len(selected_videos)}")
        logger.info(f"Total frames extracted: {total_frames}")
        logger.info(f"Frame extraction rate: {self.frame_rate}")
        logger.info(f"Minimum confidence: {self.min_confidence}")
        logger.info(f"Minimum box size: {self.min_box_size}px")
        
        for split_name, stats in split_stats.items():
            logger.info(f"\n{split_name.upper()} SPLIT:")
            logger.info(f"  Videos: {stats['processed_videos']}/{stats['videos']}")
            logger.info(f"  Frames: {stats['frames']}")
        
        logger.info(f"\nSelected videos:")
        for i, video in enumerate(selected_videos, 1):
            logger.info(f"  {i}. {video.name}")
        
        logger.info(f"\nOutput directory: {self.output_path}")
        logger.info("="*60)


def main():
    """Main function"""
    # Parameters for limited extraction
    dataset_path = "P-DESTRE"
    output_path = "limited_processed_data"
    num_videos = 15
    frame_rate = 5  # Extract every 5th frame to reduce data size
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
        return
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    print("="*70)
    print("P-DESTRE Limited Dataset Processing for YOLO Training")
    print("="*70)
    print(f"Dataset path: {dataset_path}")
    print(f"Output path: {output_path}")
    print(f"Number of videos: {num_videos}")
    print(f"Frame rate: {frame_rate} (extract every {frame_rate}th frame)")
    print(f"Min confidence: {min_confidence}")
    print(f"Min box size: {min_box_size}px")
    print(f"Train/Val/Test split: {train_ratio}/{val_ratio}/{test_ratio}")
    print("="*70)
    
    try:
        # Create extractor
        extractor = LimitedVideoExtractor(
            dataset_path=dataset_path,
            output_path=output_path,
            num_videos=num_videos,
            frame_rate=frame_rate,
            min_confidence=min_confidence,
            min_box_size=min_box_size
        )
        
        # Process limited dataset
        extractor.process_limited_dataset(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed
        )
        
        print("\n✅ Limited dataset processing completed successfully!")
        print(f"📁 Processed data saved to: {output_path}")
        print(f"📄 YOLO config file: {output_path}/config/dataset.yaml")
        print(f"📊 Dataset info: {output_path}/dataset_info.json")
        print(f"🎯 This dataset contains only {num_videos} videos for faster training!")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")


if __name__ == "__main__":
    main() 