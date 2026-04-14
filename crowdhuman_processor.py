#!/usr/bin/env python3
"""
CrowdHuman Dataset Processor for YOLO Training

Converts CrowdHuman .odgt annotations to YOLO format.
CrowdHuman annotation format (per line, JSON):
  {"ID": "image_id", "gtboxes": [{"tag": "person", "fbox": [x,y,w,h], ...}, ...]}

fbox = full body box [x_topleft, y_topleft, width, height] in pixels.
Values can be negative (person partially outside frame).
"""

import json
import cv2
import shutil
import logging
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CrowdHumanProcessor:
    """Converts CrowdHuman dataset to YOLO format."""

    def __init__(self, dataset_path: str, output_path: str,
                 min_box_size: int = 20, min_visibility: float = 0.4):
        """
        Args:
            dataset_path: Path to CrowdHuman root (contains train1/, val/, annotation_*.odgt)
            output_path:  Where to write YOLO-formatted output
            min_box_size: Minimum clipped box dimension in pixels
            min_visibility: Minimum ratio of visible area after clipping vs original area
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.min_box_size = min_box_size
        self.min_visibility = min_visibility

        self._create_dirs()

    # ------------------------------------------------------------------
    # Directory setup
    # ------------------------------------------------------------------
    def _create_dirs(self):
        for split in ('train', 'val'):
            (self.output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
        (self.output_path / 'config').mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Image lookup – images are spread across train1/ train2/ train3/ val/
    # ------------------------------------------------------------------
    def _build_image_index(self) -> Dict[str, Path]:
        """Map image ID → file path across all sub-folders."""
        index: Dict[str, Path] = {}
        for subdir in sorted(self.dataset_path.iterdir()):
            if not subdir.is_dir():
                continue
            for img_path in subdir.glob('*.jpg'):
                index[img_path.stem] = img_path
        logger.info(f"Indexed {len(index)} images across CrowdHuman sub-folders")
        return index

    # ------------------------------------------------------------------
    # Annotation parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_odgt(odgt_path: Path) -> List[dict]:
        """Read .odgt file (one JSON object per line)."""
        records = []
        with open(odgt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    # ------------------------------------------------------------------
    # Box conversion
    # ------------------------------------------------------------------
    def _fbox_to_yolo(self, fbox: List[float], img_w: int, img_h: int
                      ) -> Tuple[bool, Tuple[float, float, float, float]]:
        """
        Convert fbox [x, y, w, h] (top-left, pixels) → YOLO [cx, cy, w, h] (normalized).
        Clamps to image bounds. Returns (valid, (cx, cy, w, h)).
        """
        x, y, w, h = fbox
        orig_area = max(w * h, 1)

        # Clamp to image boundaries
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        cw = x2 - x1
        ch = y2 - y1

        # Filter: too small after clipping
        if cw < self.min_box_size or ch < self.min_box_size:
            return False, (0, 0, 0, 0)

        # Filter: too much of the box is outside the frame
        clipped_area = cw * ch
        if clipped_area / orig_area < self.min_visibility:
            return False, (0, 0, 0, 0)

        # Normalize
        cx = (x1 + cw / 2) / img_w
        cy = (y1 + ch / 2) / img_h
        nw = cw / img_w
        nh = ch / img_h

        return True, (
            max(0, min(1, cx)),
            max(0, min(1, cy)),
            max(0, min(1, nw)),
            max(0, min(1, nh)),
        )

    # ------------------------------------------------------------------
    # Process a single split
    # ------------------------------------------------------------------
    def _process_split(self, records: List[dict], image_index: Dict[str, Path],
                       split: str) -> Dict:
        stats = {'images': 0, 'boxes': 0, 'skipped_no_img': 0, 'skipped_boxes': 0}

        for rec in tqdm(records, desc=f"CrowdHuman {split}"):
            img_id = rec['ID']
            img_path = image_index.get(img_id)
            if img_path is None:
                stats['skipped_no_img'] += 1
                continue

            # Read image to get dimensions (fast – only header with IMREAD flags would
            # be ideal, but shape is needed and images vary in size)
            img = cv2.imread(str(img_path))
            if img is None:
                stats['skipped_no_img'] += 1
                continue
            img_h, img_w = img.shape[:2]

            # Convert boxes
            yolo_lines = []
            for box in rec['gtboxes']:
                if box['tag'] != 'person':
                    continue
                # Skip ignored annotations
                if box.get('extra', {}).get('ignore', 0) == 1:
                    stats['skipped_boxes'] += 1
                    continue

                valid, (cx, cy, nw, nh) = self._fbox_to_yolo(box['fbox'], img_w, img_h)
                if not valid:
                    stats['skipped_boxes'] += 1
                    continue

                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            # Skip images with no valid boxes
            if not yolo_lines:
                continue

            # Copy image
            dst_img = self.output_path / 'images' / split / f"{img_id}.jpg"
            shutil.copy2(img_path, dst_img)

            # Write label
            dst_lbl = self.output_path / 'labels' / split / f"{img_id}.txt"
            dst_lbl.write_text('\n'.join(yolo_lines) + '\n')

            stats['images'] += 1
            stats['boxes'] += len(yolo_lines)

        return stats

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def process(self, seed: int = 42):
        """Process the full CrowdHuman dataset."""
        image_index = self._build_image_index()

        all_stats = {}

        # --- Train split ---
        train_odgt = self.dataset_path / 'annotation_train.odgt'
        if train_odgt.exists():
            records = self._parse_odgt(train_odgt)
            logger.info(f"Train annotations: {len(records)} images")
            all_stats['train'] = self._process_split(records, image_index, 'train')
        else:
            logger.warning(f"Train annotation not found: {train_odgt}")

        # --- Val split ---
        val_odgt = self.dataset_path / 'annotation_val.odgt'
        if val_odgt.exists():
            records = self._parse_odgt(val_odgt)
            logger.info(f"Val annotations: {len(records)} images")
            all_stats['val'] = self._process_split(records, image_index, 'val')
        else:
            logger.warning(f"Val annotation not found: {val_odgt}")

        # Write YOLO config
        self._write_config()
        self._write_info(all_stats)
        self._print_summary(all_stats)

    # ------------------------------------------------------------------
    # Config / info files
    # ------------------------------------------------------------------
    def _write_config(self):
        import yaml
        cfg = {
            'path': str(self.output_path.resolve()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,
            'names': ['person'],
        }
        cfg_path = self.output_path / 'config' / 'dataset.yaml'
        cfg_path.write_text(yaml.dump(cfg, default_flow_style=False))
        logger.info(f"YOLO config: {cfg_path}")

    def _write_info(self, stats: Dict):
        from datetime import datetime
        info = {
            'dataset_name': 'CrowdHuman',
            'task': 'Human Detection',
            'format': 'YOLO',
            'box_type': 'fbox (full body)',
            'min_box_size': self.min_box_size,
            'min_visibility': self.min_visibility,
            'splits': stats,
            'created_date': datetime.now().isoformat(),
        }
        info_path = self.output_path / 'dataset_info.json'
        info_path.write_text(json.dumps(info, indent=2))
        logger.info(f"Dataset info: {info_path}")

    def _print_summary(self, stats: Dict):
        logger.info("\n" + "=" * 50)
        logger.info("CROWDHUMAN PROCESSING SUMMARY")
        logger.info("=" * 50)
        for split, s in stats.items():
            logger.info(f"  {split.upper()}: {s['images']} images, {s['boxes']} boxes "
                        f"(skipped {s['skipped_boxes']} boxes, {s['skipped_no_img']} missing images)")
        logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='Process CrowdHuman dataset for YOLO training')
    parser.add_argument('--dataset_path', type=str, default='../CrowdHuman',
                        help='Path to CrowdHuman root directory')
    parser.add_argument('--output_path', type=str, default='../processed_crowdhuman',
                        help='Path to save processed YOLO data')
    parser.add_argument('--min_box_size', type=int, default=20,
                        help='Minimum box dimension in pixels after clipping')
    parser.add_argument('--min_visibility', type=float, default=0.4,
                        help='Minimum visible ratio after clipping (0-1)')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    processor = CrowdHumanProcessor(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        min_box_size=args.min_box_size,
        min_visibility=args.min_visibility,
    )
    processor.process(seed=args.seed)


if __name__ == '__main__':
    main()
