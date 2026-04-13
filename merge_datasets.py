#!/usr/bin/env python3
"""
Merge processed P-DESTRE and CrowdHuman datasets into a single unified
YOLO dataset with one dataset.yaml.

Expects both datasets to already be in YOLO format (images/ + labels/ per split).
Creates symlinks (default) or copies to avoid duplicating large image files.
"""

import argparse
import json
import logging
import shutil
import yaml
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def merge_datasets(pdestre_path: str, crowdhuman_path: str, output_path: str,
                   copy: bool = False):
    """
    Merge two YOLO-formatted datasets into one.

    Args:
        pdestre_path:   Root of processed P-DESTRE YOLO data
        crowdhuman_path: Root of processed CrowdHuman YOLO data
        output_path:    Where to write the merged dataset
        copy:           If True, copy files. If False, create symlinks (saves disk).
    """
    pdestre = Path(pdestre_path)
    crowdhuman = Path(crowdhuman_path)
    out = Path(output_path)

    # P-DESTRE has train/val/test; CrowdHuman has train/val.
    # Map CrowdHuman splits to merged splits.
    split_map = {
        'pdestre':    ['train', 'val', 'test'],
        'crowdhuman': ['train', 'val'],
    }

    # Create output dirs
    for split in ('train', 'val', 'test'):
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)
    (out / 'config').mkdir(parents=True, exist_ok=True)

    transfer = shutil.copy2 if copy else _symlink

    stats = {}

    # ---- P-DESTRE ----
    for split in split_map['pdestre']:
        img_dir = pdestre / 'images' / split
        lbl_dir = pdestre / 'labels' / split
        if not img_dir.exists():
            logger.warning(f"P-DESTRE {split} images not found at {img_dir}, skipping")
            continue
        count = _transfer_split(img_dir, lbl_dir, out, split, 'pd_', transfer)
        stats[f'pdestre_{split}'] = count
        logger.info(f"P-DESTRE {split}: {count} images")

    # ---- CrowdHuman ----
    for split in split_map['crowdhuman']:
        img_dir = crowdhuman / 'images' / split
        lbl_dir = crowdhuman / 'labels' / split
        if not img_dir.exists():
            logger.warning(f"CrowdHuman {split} images not found at {img_dir}, skipping")
            continue
        count = _transfer_split(img_dir, lbl_dir, out, split, 'ch_', transfer)
        stats[f'crowdhuman_{split}'] = count
        logger.info(f"CrowdHuman {split}: {count} images")

    # ---- Write unified dataset.yaml ----
    cfg = {
        'path': str(out.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,
        'names': ['person'],
    }
    cfg_path = out / 'config' / 'dataset.yaml'
    cfg_path.write_text(yaml.dump(cfg, default_flow_style=False))
    logger.info(f"Unified YOLO config: {cfg_path}")

    # ---- Write merge info ----
    info = {
        'sources': ['P-DESTRE', 'CrowdHuman'],
        'merge_stats': stats,
        'totals': {
            'train': stats.get('pdestre_train', 0) + stats.get('crowdhuman_train', 0),
            'val': stats.get('pdestre_val', 0) + stats.get('crowdhuman_val', 0),
            'test': stats.get('pdestre_test', 0),
        },
        'method': 'copy' if copy else 'symlink',
        'created_date': datetime.now().isoformat(),
    }
    (out / 'dataset_info.json').write_text(json.dumps(info, indent=2))

    # ---- Summary ----
    logger.info("\n" + "=" * 50)
    logger.info("MERGE SUMMARY")
    logger.info("=" * 50)
    for k, v in info['totals'].items():
        logger.info(f"  {k}: {v} images")
    logger.info(f"  Total: {sum(info['totals'].values())} images")
    logger.info(f"  Config: {cfg_path}")
    logger.info("=" * 50)


def _transfer_split(img_dir: Path, lbl_dir: Path, out: Path,
                    split: str, prefix: str, transfer_fn) -> int:
    """Transfer image+label pairs from one source split into the merged output."""
    count = 0
    for img_path in sorted(img_dir.glob('*.jpg')):
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"

        # Skip images without labels
        if not lbl_path.exists():
            continue

        dst_name = f"{prefix}{stem}"
        dst_img = out / 'images' / split / f"{dst_name}.jpg"
        dst_lbl = out / 'labels' / split / f"{dst_name}.txt"

        # Skip if already exists (idempotent re-runs)
        if not dst_img.exists():
            transfer_fn(img_path, dst_img)
        if not dst_lbl.exists():
            transfer_fn(lbl_path, dst_lbl)

        count += 1
    return count


def _symlink(src: Path, dst: Path):
    """Create an absolute symlink."""
    dst.symlink_to(src.resolve())


def main():
    parser = argparse.ArgumentParser(description='Merge P-DESTRE and CrowdHuman YOLO datasets')
    parser.add_argument('--pdestre', type=str, required=True,
                        help='Path to processed P-DESTRE YOLO directory')
    parser.add_argument('--crowdhuman', type=str, required=True,
                        help='Path to processed CrowdHuman YOLO directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to write merged dataset')
    parser.add_argument('--copy', action='store_true',
                        help='Copy files instead of creating symlinks')
    args = parser.parse_args()

    merge_datasets(args.pdestre, args.crowdhuman, args.output, copy=args.copy)


if __name__ == '__main__':
    main()
