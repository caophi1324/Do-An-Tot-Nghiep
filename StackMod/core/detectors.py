"""Person and Face Detectors with crop coordinate transforms."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Sequence
import numpy as np
import cv2

from ultralytics import YOLO
from uniface.detection import RetinaFace
from uniface import Face

from .config import PersonDetectionConfig, FaceDetectionConfig


@dataclass
class PersonDetection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    conf: float
    class_id: int = 0

    @property
    def x1(self) -> float:
        return float(self.bbox[0])

    @property
    def y1(self) -> float:
        return float(self.bbox[1])

    @property
    def x2(self) -> float:
        return float(self.bbox[2])

    @property
    def y2(self) -> float:
        return float(self.bbox[3])

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


@dataclass
class FaceDetectionResult:
    bbox: np.ndarray  # [x1, y1, x2, y2] in original frame coordinates
    landmarks: np.ndarray  # (5, 2) in original frame coordinates
    conf: float

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


def offset_to_frame(face: Face, offset_x: int, offset_y: int) -> FaceDetectionResult:
    """Transforms face bounding box and 5 landmarks from crop coordinates to frame coordinates."""
    orig_box = np.asarray(face.bbox, dtype=np.float32)
    box_shifted = np.array([
        orig_box[0] + offset_x,
        orig_box[1] + offset_y,
        orig_box[2] + offset_x,
        orig_box[3] + offset_y,
    ], dtype=np.float32)

    landmarks_shifted = None
    if face.landmarks is not None:
        landmarks = np.asarray(face.landmarks, dtype=np.float32).copy()
        landmarks[:, 0] += offset_x
        landmarks[:, 1] += offset_y
        landmarks_shifted = landmarks

    conf = float(face.confidence) if face.confidence is not None else 1.0
    return FaceDetectionResult(bbox=box_shifted, landmarks=landmarks_shifted, conf=conf)


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    scaleup: bool = True,
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize and pad image with aspect ratio preservation (letterbox).
    
    Returns:
        (padded_img, ratio, (pad_w, pad_h))
    """
    shape = img.shape[:2]  # [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2.0
    dh = (new_shape[0] - new_unpad[1]) / 2.0

    if shape[::-1] != new_unpad:
        img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded_img = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded_img, r, (dw, dh)


class PersonDetector:
    """Detects people using YOLOv11n (supports both DirectML ONNX GPU and PyTorch fallback)."""

    def __init__(self, cfg: PersonDetectionConfig, providers: Optional[List[str]] = None) -> None:
        self.cfg = cfg
        self.providers = providers
        self.frame_count = 0
        self.use_onnx = False
        self.session = None
        self.last_detections: List[PersonDetection] = []

        # Check for ONNX model to run on GPU via DirectML
        onnx_candidates = [
            cfg.model_path.replace(".pt", ".onnx"),
            os.path.join("models", "yolo11n.onnx"),
            "yolo11n.onnx",
        ]
        chosen_onnx = next((p for p in onnx_candidates if os.path.exists(p)), None)

        if chosen_onnx and providers:
            try:
                import onnxruntime as ort
                from .config import get_dml_session_options, get_cuda_session_options
                device_id = getattr(cfg, "device_id", 0)
                if any("Dml" in str(p) for p in providers):
                    sess_opts, prov_opts = get_dml_session_options(device_id=device_id)
                    self.session = ort.InferenceSession(
                        chosen_onnx,
                        sess_options=sess_opts,
                        providers=providers,
                        provider_options=prov_opts if len(prov_opts) == len(providers) else None,
                    )
                elif any("CUDA" in str(p) for p in providers):
                    sess_opts, prov_opts = get_cuda_session_options(device_id=device_id)
                    self.session = ort.InferenceSession(
                        chosen_onnx,
                        sess_options=sess_opts,
                        providers=providers,
                        provider_options=prov_opts if len(prov_opts) == len(providers) else None,
                    )
                else:
                    self.session = ort.InferenceSession(chosen_onnx, providers=providers)
                self.input_name = self.session.get_inputs()[0].name
                self.use_onnx = True
            except Exception as e:
                self.use_onnx = False

        if not self.use_onnx:
            model_path = cfg.model_path
            if not os.path.exists(model_path):
                alt_path = os.path.join("models", os.path.basename(model_path))
                if os.path.exists(alt_path):
                    model_path = alt_path
            self.model = YOLO(model_path)

    def update_config(self, cfg: PersonDetectionConfig) -> None:
        """Dynamically applies updated person detection configuration."""
        self.cfg = cfg

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        if not self.cfg.enabled or frame is None or frame.size == 0:
            self.last_detections = []
            return []

        self.frame_count += 1
        # Honor skip_frames: reuse last detections on skipped frames
        if self.cfg.skip_frames > 0 and (self.frame_count % (self.cfg.skip_frames + 1) != 1):
            return self.last_detections

        h_orig, w_orig = frame.shape[:2]

        # 1. DirectML ONNX Fast Path (runs on GPU)
        if self.use_onnx and self.session is not None:
            # Letterbox preserving 16:9 aspect ratio without distortion
            img_padded, ratio, (dw, dh) = letterbox(frame, (640, 640))
            img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
            blob = np.transpose(img_rgb, (2, 0, 1))[None, :, :, :].astype(np.float32) / 255.0

            if self.cfg.half:
                try:
                    input_type = self.session.get_inputs()[0].type
                    if "float16" in input_type:
                        blob = blob.astype(np.float16)
                except Exception:
                    pass

            out = self.session.run(None, {self.input_name: blob})[0]
            preds = np.transpose(out[0], (1, 0))  # (8400, 84)

            # Class index: person_class_id offset by 4 (cx, cy, w, h)
            cls_col = 4 + int(self.cfg.person_class_id)
            if cls_col < preds.shape[1]:
                person_scores = preds[:, cls_col]
            else:
                person_scores = preds[:, 4]

            mask = person_scores >= self.cfg.conf
            filtered_preds = preds[mask]
            filtered_scores = person_scores[mask]

            if len(filtered_scores) == 0:
                self.last_detections = []
                return []

            boxes = []
            for p in filtered_preds:
                cx, cy, w, h = p[:4]
                # Unpad and scale back to original frame coordinates
                x1 = (cx - w / 2.0 - dw) / ratio
                y1 = (cy - h / 2.0 - dh) / ratio
                x2 = (cx + w / 2.0 - dw) / ratio
                y2 = (cy + h / 2.0 - dh) / ratio

                # Clip coordinates strictly within original frame dimensions
                x1 = max(0.0, min(float(w_orig), float(x1)))
                y1 = max(0.0, min(float(h_orig), float(y1)))
                x2 = max(0.0, min(float(w_orig), float(x2)))
                y2 = max(0.0, min(float(h_orig), float(y2)))

                if x2 > x1 and y2 > y1:
                    boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

            indices = cv2.dnn.NMSBoxes(boxes, filtered_scores.tolist(), self.cfg.conf, self.cfg.iou)
            detections: List[PersonDetection] = []
            if len(indices) > 0:
                for idx in indices.flatten():
                    bx, by, bw, bh = boxes[idx]
                    score = float(filtered_scores[idx])
                    detections.append(PersonDetection(
                        bbox=np.array([bx, by, bx + bw, by + bh], dtype=np.float32),
                        conf=score,
                        class_id=self.cfg.person_class_id,
                    ))
            self.last_detections = detections
            return detections

        # 2. PyTorch Fallback Path
        predict_kwargs = {
            "classes": [self.cfg.person_class_id],
            "conf": self.cfg.conf,
            "iou": self.cfg.iou,
            "verbose": False,
        }
        if self.cfg.half:
            predict_kwargs["half"] = True

        # Support CUDA/CPU device routing in PyTorch fallback
        if self.providers:
            if any("cuda" in str(p).lower() for p in self.providers):
                predict_kwargs["device"] = getattr(self.cfg, "device_id", 0)
            elif all("cpu" in str(p).lower() for p in self.providers):
                predict_kwargs["device"] = "cpu"

        results = self.model.predict(frame, **predict_kwargs)

        detections: List[PersonDetection] = []
        if not results:
            self.last_detections = []
            return detections

        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes_xyxy, confs):
                box_clipped = np.array([
                    max(0.0, min(float(w_orig), float(box[0]))),
                    max(0.0, min(float(h_orig), float(box[1]))),
                    max(0.0, min(float(w_orig), float(box[2]))),
                    max(0.0, min(float(h_orig), float(box[3]))),
                ], dtype=np.float32)
                detections.append(PersonDetection(
                    bbox=box_clipped,
                    conf=float(conf),
                    class_id=self.cfg.person_class_id,
                ))

        self.last_detections = detections
        return detections


try:
    from uniface.constants import RetinaFaceWeights
except ImportError:
    RetinaFaceWeights = None


def get_retinaface_weight(name: Any) -> Any:
    if RetinaFaceWeights is None or isinstance(name, RetinaFaceWeights):
        return name
    name_str = str(name).lower()
    for member in RetinaFaceWeights:
        if member.value.lower() == name_str or member.name.lower() == name_str:
            return member
    return RetinaFaceWeights.MNET_V2


class FaceDetector:
    """Detects faces using RetinaFace with crop support."""

    def __init__(self, cfg: FaceDetectionConfig, providers: Optional[List[str]] = None) -> None:
        self.cfg = cfg
        self.providers = providers
        model_name = get_retinaface_weight(getattr(cfg, "model_name", "retinaface_mnet_v2"))
        try:
            self.model = RetinaFace(
                model_name=model_name,
                confidence_threshold=cfg.conf,
                nms_threshold=cfg.nms,
                providers=providers,
            )
        except Exception:
            # Fallback to CPU if provider fails
            self.model = RetinaFace(
                model_name=model_name,
                confidence_threshold=cfg.conf,
                nms_threshold=cfg.nms,
                providers=["CPUExecutionProvider"],
            )

    def update_config(self, cfg: FaceDetectionConfig) -> None:
        """Dynamically applies updated face detection configuration."""
        self.cfg = cfg
        if hasattr(self.model, "confidence_threshold"):
            self.model.confidence_threshold = cfg.conf
        if hasattr(self.model, "nms_threshold"):
            self.model.nms_threshold = cfg.nms

    def detect_in_frame(self, frame: np.ndarray) -> List[FaceDetectionResult]:
        """Runs face detection directly on the full frame."""
        if not self.cfg.enabled or frame is None or frame.size == 0:
            return []

        # Ensure runtime threshold matches current config
        if hasattr(self.model, "confidence_threshold"):
            self.model.confidence_threshold = self.cfg.conf
        if hasattr(self.model, "nms_threshold"):
            self.model.nms_threshold = self.cfg.nms

        raw_faces = self.model.detect(frame)
        results: List[FaceDetectionResult] = []
        for face in raw_faces:
            box = np.asarray(face.bbox, dtype=np.float32)
            # Filter by min_face_size
            w = box[2] - box[0]
            h = box[3] - box[1]
            if min(w, h) < self.cfg.min_face_size:
                continue

            conf = float(face.confidence) if face.confidence is not None else 1.0
            if conf < self.cfg.conf:
                continue

            landmarks = np.asarray(face.landmarks, dtype=np.float32) if face.landmarks is not None else None
            results.append(FaceDetectionResult(bbox=box, landmarks=landmarks, conf=conf))
        return results

    def detect_in_person_crop(
        self, frame: np.ndarray, person_box: Sequence[float] | np.ndarray
    ) -> Optional[FaceDetectionResult]:
        """Detects face inside the upper region (e.g. upper 60%) of a person bounding box."""
        if not self.cfg.enabled or frame is None or frame.size == 0:
            return None

        # Ensure runtime threshold matches current config
        if hasattr(self.model, "confidence_threshold"):
            self.model.confidence_threshold = self.cfg.conf
        if hasattr(self.model, "nms_threshold"):
            self.model.nms_threshold = self.cfg.nms

        h_img, w_img = frame.shape[:2]
        px1 = max(0, int(person_box[0]))
        py1 = max(0, int(person_box[1]))
        px2 = min(w_img, int(person_box[2]))
        py2 = min(h_img, int(person_box[3]))

        pw = px2 - px1
        ph = py2 - py1
        if pw <= 10 or ph <= 10:
            return None

        # Contextual padding: expand upper crop by 10% above head, 12% on sides,
        # and ensure sufficient depth for chin/mouth landmarks
        pad_top = int(0.10 * ph)
        pad_side = int(0.12 * pw)
        pad_bottom = int(0.05 * ph)

        crop_x1 = max(0, px1 - pad_side)
        crop_y1 = max(0, py1 - pad_top)
        crop_x2 = min(w_img, px2 + pad_side)
        crop_y2 = min(h_img, int(py1 + max(0.65, self.cfg.crop_upper_ratio) * ph + pad_bottom))

        if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
            return None
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        if crop.size == 0:
            return None

        faces = self.model.detect(crop)
        if not faces:
            return None

        # Filter by min_face_size and conf
        valid_faces = [
            f for f in faces
            if (f.confidence is None or f.confidence >= self.cfg.conf)
            and min(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]) >= self.cfg.min_face_size
        ]
        if not valid_faces:
            return None

        # Pick largest face in crop
        largest_face = max(
            valid_faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        res = offset_to_frame(largest_face, crop_x1, crop_y1)
        res.bbox[0] = max(0.0, float(res.bbox[0]))
        res.bbox[1] = max(0.0, float(res.bbox[1]))
        res.bbox[2] = min(float(w_img), float(res.bbox[2]))
        res.bbox[3] = min(float(h_img), float(res.bbox[3]))
        return res
