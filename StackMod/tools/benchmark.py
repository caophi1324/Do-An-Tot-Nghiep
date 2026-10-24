"""Benchmark and Load Testing script measuring latency, throughput (FPS), and resource usage.

Supports both synthetic frame benchmarks and real test video files with optional output recording.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional
import numpy as np
import cv2
import psutil

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import AppConfig, ExecutionProvider, get_optimal_providers
from core.pipeline import Pipeline, render_overlay


def run_benchmark(
    video_path: Optional[str] = None,
    num_frames: Optional[int] = None,
    device: str = "auto",
    resolution: tuple = (1280, 720),
    render: bool = True,
    save_output: Optional[str] = None,
) -> None:
    print("=" * 70)
    print("🚀 AI FACE RECOGNITION & TRACKING - BENCHMARK & LOAD TEST")
    print("=" * 70)

    # Initialize configuration
    cfg = AppConfig()
    if device.lower() in ("directml", "dml"):
        cfg.hardware.provider = ExecutionProvider.DIRECTML
    elif device.lower() == "cuda":
        cfg.hardware.provider = ExecutionProvider.CUDA
    elif device.lower() == "cpu":
        cfg.hardware.provider = ExecutionProvider.CPU
    else:
        cfg.hardware.provider = ExecutionProvider.AUTO

    providers = get_optimal_providers(cfg.hardware.provider)
    print(f"[*] Hardware Provider Selected: {providers}")
    print(f"[*] CPU Cores: {psutil.cpu_count(logical=True)} | RAM Available: {psutil.virtual_memory().available / (1024**3):.1f} GB")

    print("[*] Initializing Models & Pipeline...")
    pipeline = Pipeline(cfg)

    # Video Source Setup
    cap: Optional[cv2.VideoCapture] = None
    if video_path:
        if not os.path.exists(video_path):
            print(f"[!] Error: Video file not found: {video_path}")
            return
        cap = cv2.VideoCapture(video_path)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[*] Source: Video File [{video_path}]")
        print(f"[*] Resolution: {vw}x{vh} | Native FPS: {video_fps:.1f} | Total Frames: {total_video_frames}")
        target_frames = num_frames if num_frames is not None else total_video_frames
    else:
        w, h = resolution
        print(f"[*] Source: Synthetic Scene ({w}x{h})")
        target_frames = num_frames if num_frames is not None else 60

    # Video Writer if save_output is requested
    writer: Optional[cv2.VideoWriter] = None
    if save_output:
        os.makedirs(os.path.dirname(os.path.abspath(save_output)), exist_ok=True)
        out_w, out_h = (vw, vh) if cap is not None else resolution
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_output, fourcc, 30.0, (out_w, out_h))
        print(f"[*] Output video recording enabled: {save_output}")

    # Warm-up run (3 frames)
    print("[*] Warming up pipeline...")
    if cap is not None:
        ret, warm_frame = cap.read()
        if ret and warm_frame is not None:
            for _ in range(3):
                _ = pipeline.process(warm_frame, time.time())
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind
    else:
        dummy = np.full((h, w, 3), 120, dtype=np.uint8)
        for _ in range(3):
            _ = pipeline.process(dummy, time.time())

    # Benchmark loop
    print(f"\n[*] Starting Load Test ({target_frames} frames)...")
    latencies_ms = []
    process_start = time.perf_counter()
    proc = psutil.Process()
    initial_mem_mb = proc.memory_info().rss / (1024 * 1024)

    total_persons_detected = 0
    total_faces_tracked = 0
    frames_read = 0

    try:
        while frames_read < target_frames:
            if cap is not None:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
            else:
                # Generate synthetic test frame with realistic box shapes
                frame = np.full((h, w, 3), 120, dtype=np.uint8)
                cv2.rectangle(frame, (w // 2 - 100, h // 2 - 200), (w // 2 + 100, h // 2 + 250), (180, 150, 120), -1)
                cv2.circle(frame, (w // 2, h // 2 - 120), 50, (210, 180, 140), -1)

            ts = time.time()
            t0 = time.perf_counter()

            # Pipeline execution
            results = pipeline.process(frame, timestamp=ts)

            # Optional overlay rendering
            if render:
                rendered = render_overlay(frame, results, cfg.overlay)
                if writer is not None:
                    writer.write(rendered)

            t1 = time.perf_counter()
            frame_time_ms = (t1 - t0) * 1000.0
            latencies_ms.append(frame_time_ms)

            total_persons_detected += len(results)
            total_faces_tracked += sum(1 for r in results if r.face_bbox is not None)
            frames_read += 1

            if frames_read % 30 == 0 or frames_read == target_frames:
                instant_fps = 1000.0 / frame_time_ms if frame_time_ms > 0 else 0
                print(f"    Frame {frames_read}/{target_frames} | Latency: {frame_time_ms:.1f}ms | Current FPS: {instant_fps:.1f}")

    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()

    total_wall_time = time.perf_counter() - process_start
    final_mem_mb = proc.memory_info().rss / (1024 * 1024)
    avg_fps = frames_read / max(1e-4, total_wall_time)

    # Detailed statistics
    lat_arr = np.array(latencies_ms)
    mean_lat = np.mean(lat_arr)
    std_lat = np.std(lat_arr)
    min_lat = np.min(lat_arr)
    max_lat = np.max(lat_arr)
    p50_lat = np.percentile(lat_arr, 50)
    p90_lat = np.percentile(lat_arr, 90)
    p95_lat = np.percentile(lat_arr, 95)
    p99_lat = np.percentile(lat_arr, 99)

    print("\n" + "=" * 70)
    print("📊 LOAD TEST & BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  • Total Frames Processed:   {frames_read} frames")
    print(f"  • Total Execution Time:     {total_wall_time:.2f} seconds")
    print(f"  • Overall Throughput:       {avg_fps:.1f} FPS")
    print(f"  • Average Latency:          {mean_lat:.2f} ms (± {std_lat:.2f} ms)")
    print(f"  • Latency Percentiles:")
    print(f"      - Min / Max:            {min_lat:.2f} ms / {max_lat:.2f} ms")
    print(f"      - 50th percentile (P50): {p50_lat:.2f} ms")
    print(f"      - 90th percentile (P90): {p90_lat:.2f} ms")
    print(f"      - 95th percentile (P95): {p95_lat:.2f} ms")
    print(f"      - 99th percentile (P99): {p99_lat:.2f} ms")
    print(f"  • Resource Utilization:")
    print(f"      - RAM Footprint:        {final_mem_mb:.1f} MB (Delta: {final_mem_mb - initial_mem_mb:+.1f} MB)")
    print(f"      - Persons Tracked:      {total_persons_detected} detections")
    print(f"      - Faces Analyzed:       {total_faces_tracked} detections")
    if save_output:
        print(f"  • Output Video Saved:       {save_output}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="AI Face Recognition & Tracking Benchmark / Load Test Tool")
    parser.add_argument("--video", type=str, default=None, help="Path to video file for load testing (e.g. data/test.mp4)")
    parser.add_argument("--frames", type=int, default=None, help="Maximum number of frames to evaluate")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "directml", "cuda", "cpu"], help="Hardware acceleration provider")
    parser.add_argument("--no-render", action="store_true", help="Skip overlay rendering for pure model inference benchmarking")
    parser.add_argument("--save-output", type=str, default=None, help="Path to save annotated MP4 video output")
    parser.add_argument("--width", type=int, default=1280, help="Synthetic frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Synthetic frame height (default: 720)")

    args = parser.parse_args()

    run_benchmark(
        video_path=args.video,
        num_frames=args.frames,
        device=args.device,
        resolution=(args.width, args.height),
        render=not args.no_render,
        save_output=args.save_output,
    )


if __name__ == "__main__":
    main()
