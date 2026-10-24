"""Pre-downloads and verifies all required neural network models for offline deployments."""
from __future__ import annotations

import os
import sys
import shutil

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from uniface.detection import RetinaFace
from uniface.quality import EDifFIQA
from uniface.recognition import EdgeFace
from uniface import get_cache_dir


def download_all_models(target_models_dir: str = "models") -> None:
    target_abs = os.path.abspath(target_models_dir)
    os.makedirs(target_abs, exist_ok=True)
    import uniface
    uniface.set_cache_dir(target_abs)

    print("=" * 60)
    print("Downloading & verifying AI models...")
    print("=" * 60)

    # 1. YOLOv11n
    print("[1/4] Loading YOLOv11n...")
    yolo = YOLO("yolo11n.pt")
    if os.path.exists("yolo11n.pt") and not os.path.exists(os.path.join(target_models_dir, "yolo11n.pt")):
        shutil.copy("yolo11n.pt", os.path.join(target_models_dir, "yolo11n.pt"))
    print("✓ YOLOv11n ready.")

    # 2. RetinaFace
    print("[2/4] Loading RetinaFace (MNet V2)...")
    rf = RetinaFace()
    print("✓ RetinaFace ready.")

    # 3. eDifFIQA-T
    print("[3/4] Loading eDifFIQA-T Quality Estimator...")
    qa = EDifFIQA()
    print("✓ eDifFIQA-T ready.")

    # 4. EdgeFace
    print("[4/4] Loading EdgeFace (XXS)...")
    ef = EdgeFace()
    print("✓ EdgeFace ready.")

    cache_dir = get_cache_dir()
    print("=" * 60)
    print(f"All models cached successfully in:\n  UniFace Cache: {cache_dir}\n  App Models Dir: {os.path.abspath(target_models_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    download_all_models()
