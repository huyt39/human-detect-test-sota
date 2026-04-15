"""
GPU-accelerated P-DESTRE data processor.

Optimizations over data_processor.py:
  1. decord GPU backend for hardware video decoding (NVDEC)
  2. Batch frame reading (grab N frames at once instead of sequential seek)
  3. Concurrent video processing with ProcessPoolExecutor
  4. turbojpeg for faster JPEG encoding (~3-5x vs cv2.imwrite)
  5. Vectorised YOLO annotation conversion (numpy, no per-row loop)
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Conditional imports: fall back gracefully so the script still works on a
# CPU-only machine (just slower).
# ---------------------------------------------------------------------------
try:
    import decord
    from decord import VideoReader, gpu, cpu

    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False

try:
    from turbojpeg import TurboJPEG

    _tj = TurboJPEG()
    HAS_TURBOJPEG = True
except ImportError:
    HAS_TURBOJPEG = False

if not HAS_TURBOJPEG:
    import cv2  # fallback for JPEG encoding

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# P-DESTRE annotation columns (only first 7 used for detection)
ANNOTATION_COLUMNS = [
    "frame_id", "track_id", "x", "y", "width", "height", "confidence",
    "yaw", "pitch", "roll",
    *[f"attr_{i}" for i in range(16)],
]

VIDEO_EXTENSIONS = {".mp4", ".MP4", ".avi", ".mov"}


# ===================================================================
# Helper functions (module-level so they are pickle-able for multiprocessing)
# ===================================================================

def _parse_annotations(annotation_file: Path, min_confidence: float,
                       min_box_size: int) -> pd.DataFrame:
    """Parse and filter a P-DESTRE annotation file."""
    try:
        df = pd.read_csv(annotation_file, header=None)
        df.columns = ANNOTATION_COLUMNS[: len(df.columns)]

        mask = (
            (df["confidence"] >= min_confidence)
            & (df["width"] >= min_box_size)
            & (df["height"] >= min_box_size)
            & (df["track_id"] != -1)
            & (df["x"] >= 0)
            & (df["y"] >= 0)
            & (df["width"] > 0)
            & (df["height"] > 0)
        )
        return df.loc[mask].copy()
    except Exception as e:
        logger.error(f"Error parsing {annotation_file}: {e}")
        return pd.DataFrame()


def _annotations_to_yolo(frame_annots: pd.DataFrame,
                         img_w: int, img_h: int) -> str:
    """Vectorised conversion of a frame's annotations to YOLO text lines."""
    x = frame_annots["x"].values
    y = frame_annots["y"].values
    w = frame_annots["width"].values
    h = frame_annots["height"].values

    xc = np.clip((x + w / 2) / img_w, 0, 1)
    yc = np.clip((y + h / 2) / img_h, 0, 1)
    wn = np.clip(w / img_w, 0, 1)
    hn = np.clip(h / img_h, 0, 1)

    lines = [
        f"0 {xc[i]:.6f} {yc[i]:.6f} {wn[i]:.6f} {hn[i]:.6f}"
        for i in range(len(xc))
    ]
    return "\n".join(lines) + "\n" if lines else ""


def _encode_jpeg(frame_rgb: np.ndarray, quality: int = 95) -> bytes:
    """Encode a RGB numpy array to JPEG bytes (GPU-friendly path)."""
    if HAS_TURBOJPEG:
        # turbojpeg expects BGR by default; convert from RGB
        frame_bgr = frame_rgb[:, :, ::-1].copy()
        return _tj.encode(frame_bgr, quality=quality)
    else:
        frame_bgr = frame_rgb[:, :, ::-1]
        _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()


def _process_single_video(
    video_path: str,
    annotation_path: str,
    output_root: str,
    split: str,
    frame_rate: int,
    min_confidence: float,
    min_box_size: int,
    use_gpu: bool,
    gpu_id: int,
) -> int:
    """Process one video file — designed to run in a worker process."""
    video_file = Path(video_path)
    annotation_file = Path(annotation_path)
    output_path = Path(output_root)

    # Parse annotations
    annots = _parse_annotations(annotation_file, min_confidence, min_box_size)
    if annots.empty:
        return 0

    # Build a set of annotated frame IDs (1-based) that we actually need
    needed_frame_ids = set(annots["frame_id"].unique())

    # ----- Open video with decord (GPU) or fallback -----
    if HAS_DECORD and use_gpu:
        try:
            ctx = gpu(gpu_id)
            vr = VideoReader(str(video_file), ctx=ctx)
        except Exception:
            # Some codecs don't work on GPU; fall back to CPU decord
            ctx = cpu(0)
            vr = VideoReader(str(video_file), ctx=ctx)
    elif HAS_DECORD:
        vr = VideoReader(str(video_file), ctx=cpu(0))
    else:
        vr = None

    if vr is not None:
        total_frames = len(vr)
        img_w, img_h = vr[0].shape[1], vr[0].shape[0]
    else:
        # cv2 fallback
        import cv2 as _cv2

        cap = _cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            return 0
        total_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
        img_w = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))

    # Pre-compute which 0-based frame indices to extract
    indices_to_extract = []
    for idx in range(0, total_frames, frame_rate):
        pdestre_frame_id = idx + 1  # 1-based
        if pdestre_frame_id in needed_frame_ids:
            indices_to_extract.append(idx)

    if not indices_to_extract:
        if vr is None:
            cap.release()
        return 0

    video_name = video_file.stem
    img_dir = output_path / "images" / split
    lbl_dir = output_path / "labels" / split
    extracted = 0

    if vr is not None:
        # --- Batch read with decord (much faster than sequential seeks) ---
        BATCH = 64
        for batch_start in range(0, len(indices_to_extract), BATCH):
            batch_indices = indices_to_extract[batch_start : batch_start + BATCH]
            # decord batch read: returns ndarray (N, H, W, 3) in RGB
            frames = vr.get_batch(batch_indices).asnumpy()

            for i, idx in enumerate(batch_indices):
                frame_rgb = frames[i]
                pdestre_id = idx + 1
                frame_annots = annots[annots["frame_id"] == pdestre_id]

                fname = f"{video_name}_frame_{idx:06d}"
                # JPEG encode
                jpg_bytes = _encode_jpeg(frame_rgb)
                (img_dir / f"{fname}.jpg").write_bytes(jpg_bytes)
                # YOLO label
                yolo_txt = _annotations_to_yolo(frame_annots, img_w, img_h)
                (lbl_dir / f"{fname}.txt").write_text(yolo_txt)
                extracted += 1
    else:
        # --- cv2 fallback (sequential) ---
        import cv2 as _cv2

        extract_set = set(indices_to_extract)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in extract_set:
                pdestre_id = frame_idx + 1
                frame_annots = annots[annots["frame_id"] == pdestre_id]
                fname = f"{video_name}_frame_{frame_idx:06d}"
                _cv2.imwrite(
                    str(img_dir / f"{fname}.jpg"), frame,
                    [_cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                yolo_txt = _annotations_to_yolo(frame_annots, img_w, img_h)
                (lbl_dir / f"{fname}.txt").write_text(yolo_txt)
                extracted += 1
            frame_idx += 1
        cap.release()

    return extracted


# ===================================================================
# Main processor class
# ===================================================================

class PDestreDataProcessorGPU:
    """GPU-accelerated P-DESTRE data processor for YOLO training data."""

    def __init__(
        self,
        dataset_path: str,
        output_path: str,
        frame_rate: int = 10,
        min_confidence: float = 0.5,
        min_box_size: int = 20,
        gpu_id: int = 0,
        max_workers: int = 4,
    ):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.frame_rate = frame_rate
        self.min_confidence = min_confidence
        self.min_box_size = min_box_size
        self.gpu_id = gpu_id
        self.max_workers = max_workers

        # Detect GPU availability
        self.use_gpu = HAS_DECORD and self._check_gpu()

        self._create_dirs()
        self._log_backend_info()

    # ----- setup helpers -----

    @staticmethod
    def _check_gpu() -> bool:
        try:
            decord.bridge.set_bridge("native")
            # Try to create a GPU context — fails if no NVIDIA driver
            _ = gpu(0)
            return True
        except Exception:
            return False

    def _create_dirs(self):
        for sub in ("images", "labels"):
            for split in ("train", "val", "test"):
                (self.output_path / sub / split).mkdir(parents=True, exist_ok=True)
        (self.output_path / "config").mkdir(parents=True, exist_ok=True)

    def _log_backend_info(self):
        logger.info(f"decord available: {HAS_DECORD}")
        logger.info(f"turbojpeg available: {HAS_TURBOJPEG}")
        logger.info(f"GPU decoding: {'YES' if self.use_gpu else 'NO (CPU fallback)'}")
        logger.info(f"Max parallel workers: {self.max_workers}")

    # ----- main entry -----

    def process_dataset(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")

        random.seed(seed)

        video_files = self._get_video_files()
        random.shuffle(video_files)

        n = len(video_files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": video_files[:n_train],
            "val": video_files[n_train : n_train + n_val],
            "test": video_files[n_train + n_val :],
        }
        logger.info(
            f"Split: train={len(splits['train'])}, "
            f"val={len(splits['val'])}, test={len(splits['test'])}"
        )

        total_frames = 0
        split_stats: Dict[str, dict] = {}

        for split_name, videos in splits.items():
            logger.info(f"\n--- Processing {split_name} split ({len(videos)} videos) ---")
            frames = self._process_split(split_name, videos)
            split_stats[split_name] = {
                "videos": len(videos),
                "frames": frames,
            }
            total_frames += frames

        self._write_yolo_config()
        self._write_dataset_info(split_stats, total_frames)
        self._print_summary(split_stats, total_frames)

    # ----- split processing with parallelism -----

    def _process_split(self, split: str, videos: List[Path]) -> int:
        """Process all videos for one split using parallel workers."""
        # Build (video, annotation) pairs
        tasks = []
        for vf in videos:
            af = self.dataset_path / "annotation" / f"{vf.stem}.txt"
            if af.exists():
                tasks.append((vf, af))
            else:
                logger.warning(f"Missing annotation: {af}")

        if not tasks:
            return 0

        total_frames = 0
        # Use ProcessPoolExecutor for true parallelism (GIL-free)
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}
            for vf, af in tasks:
                fut = pool.submit(
                    _process_single_video,
                    str(vf),
                    str(af),
                    str(self.output_path),
                    split,
                    self.frame_rate,
                    self.min_confidence,
                    self.min_box_size,
                    self.use_gpu,
                    self.gpu_id,
                )
                futures[fut] = vf.name

            pbar = tqdm(total=len(futures), desc=f"{split}")
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    n = fut.result()
                    total_frames += n
                    pbar.set_postfix(last=name, frames=n)
                except Exception as e:
                    logger.error(f"Failed {name}: {e}")
                pbar.update(1)
            pbar.close()

        return total_frames

    # ----- helpers -----

    def _get_video_files(self) -> List[Path]:
        vdir = self.dataset_path / "videos"
        if not vdir.exists():
            raise FileNotFoundError(f"Videos dir not found: {vdir}")
        files = [f for f in vdir.iterdir() if f.suffix in VIDEO_EXTENSIONS]
        logger.info(f"Found {len(files)} video files")
        return files

    def _write_yolo_config(self):
        cfg = {
            "path": str(self.output_path),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": 1,
            "names": ["person"],
        }
        out = self.output_path / "config" / "dataset.yaml"
        with open(out, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        logger.info(f"Wrote YOLO config: {out}")

    def _write_dataset_info(self, split_stats: Dict, total_frames: int):
        info = {
            "dataset_name": "P-DESTRE",
            "task": "Human Detection",
            "format": "YOLO",
            "total_frames": total_frames,
            "processing": {
                "frame_rate": self.frame_rate,
                "min_confidence": self.min_confidence,
                "min_box_size": self.min_box_size,
                "gpu_decode": self.use_gpu,
                "max_workers": self.max_workers,
            },
            "splits": split_stats,
            "created_date": pd.Timestamp.now().isoformat(),
        }
        out = self.output_path / "dataset_info.json"
        with open(out, "w") as f:
            json.dump(info, f, indent=2)
        logger.info(f"Wrote dataset info: {out}")

    def _print_summary(self, split_stats: Dict, total_frames: int):
        logger.info("\n" + "=" * 50)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Backend: {'GPU (NVDEC)' if self.use_gpu else 'CPU'}")
        logger.info(f"Total frames: {total_frames}")
        for name, s in split_stats.items():
            logger.info(f"  {name}: {s['frames']} frames from {s['videos']} videos")
        logger.info(f"Output: {self.output_path}")
        logger.info("=" * 50)


# ===================================================================
# CLI
# ===================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GPU-accelerated P-DESTRE → YOLO processor"
    )
    parser.add_argument("--dataset_path", default="../P-DESTRE")
    parser.add_argument("--output_path", default="../processed_pdestre")
    parser.add_argument("--frame_rate", type=int, default=10)
    parser.add_argument("--min_confidence", type=float, default=0.5)
    parser.add_argument("--min_box_size", type=int, default=20)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel video workers (default: 4)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    processor = PDestreDataProcessorGPU(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        frame_rate=args.frame_rate,
        min_confidence=args.min_confidence,
        min_box_size=args.min_box_size,
        gpu_id=args.gpu_id,
        max_workers=args.workers,
    )

    processor.process_dataset(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    logger.info("Done!")


if __name__ == "__main__":
    main()
