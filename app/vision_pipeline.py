import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.models import EvaluationMetadata


TARGET_FPS = float(os.getenv("EVAL_TARGET_FPS", "10.0"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "416"))
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.25"))
CLASSIFIER_IMAGE_SIZE = (224, 224)
BYPASS_VEHICLE_TYPES = {"truck", "motorcycle", "bus"}
SUPPORTED_VEHICLE_TYPES = {"car", "truck", "motorcycle", "bus"}
DEFAULT_YOLO_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

os.environ.setdefault("YOLO_AUTOINSTALL", "false")


@dataclass(frozen=True)
class ClassificationLabel:
    brand: str | None


@dataclass(frozen=True)
class Labels:
    yolo: dict[int, str]
    classifier: dict[int, ClassificationLabel]


Point = tuple[float, float]
RoiMap = dict[str, list[Point]]


def run_video_pipeline(
    db: Session,
    submission_id: int,
    cctv_id: str,
    video_path: str | Path,
    yolo_model_path: str | Path,
    classifier_model_path: str | Path | None,
    labels_json_path: str | Path | None,
    roi_json_path: str | Path | None = None,
) -> int:
    cv2 = _import_cv2()
    yolo_model = _load_yolo(yolo_model_path)
    classifier_session = (
        _load_classifier_session(classifier_model_path)
        if classifier_model_path is not None
        else None
    )
    labels = load_labels(labels_json_path)
    roi_polygon = load_roi_polygon(roi_json_path, cctv_id)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open input video: {video_path}")

    db.execute(
        delete(EvaluationMetadata)
        .where(EvaluationMetadata.submission_id == submission_id)
        .where(EvaluationMetadata.cctv_id == cctv_id)
    )
    db.commit()

    metadata_rows: list[dict[str, Any]] = []
    track_frame_counts: dict[int, int] = {}
    track_attributes: dict[int, ClassificationLabel] = {}
    source_fps = capture.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    frame_index = -1
    next_sample_time = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_index += 1
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if timestamp <= 0 and source_fps > 0:
                timestamp = frame_index / source_fps

            if timestamp + 1e-9 < next_sample_time:
                continue
            next_sample_time += 1.0 / TARGET_FPS

            result = yolo_model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                device="cpu",
                imgsz=YOLO_IMGSZ,
                conf=YOLO_CONF,
                verbose=False,
            )[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.id is None:
                continue

            xyxy_rows = _to_numpy(boxes.xyxy)
            class_ids = _to_numpy(boxes.cls).astype(int)
            track_ids = _to_numpy(boxes.id).astype(int)
            names = getattr(result, "names", None) or getattr(yolo_model, "names", {})

            for xyxy, class_id, track_id in zip(xyxy_rows, class_ids, track_ids, strict=False):
                vehicle_type = normalize_vehicle_type(resolve_yolo_label(class_id, labels, names))
                if vehicle_type not in SUPPORTED_VEHICLE_TYPES:
                    continue

                x1, y1, x2, y2 = [float(value) for value in xyxy]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                if roi_polygon and not point_in_polygon(center_x, center_y, roi_polygon):
                    continue

                brand = None

                if vehicle_type == "car" and classifier_session is not None:
                    track_frame_counts[track_id] = track_frame_counts.get(track_id, 0) + 1
                    if track_id not in track_attributes and track_frame_counts[track_id] >= 5:
                        crop = crop_frame(frame, x1, y1, x2, y2)
                        if crop is not None:
                            track_attributes[track_id] = classify_vehicle(
                                classifier_session=classifier_session,
                                crop=crop,
                                labels=labels,
                            )

                    if track_id in track_attributes:
                        brand = track_attributes[track_id].brand

                metadata_rows.append(
                    {
                        "submission_id": submission_id,
                        "cctv_id": cctv_id,
                        "track_id": int(track_id),
                        "timestamp": float(timestamp),
                        "vehicle_type": vehicle_type,
                        "brand": brand,
                        "color": None,
                        "bbox_x": x1,
                        "bbox_y": y1,
                        "bbox_w": max(0.0, x2 - x1),
                        "bbox_h": max(0.0, y2 - y1),
                    }
                )
    finally:
        capture.release()

    for row in metadata_rows:
        if row["vehicle_type"] == "car" and row["track_id"] in track_attributes:
            label = track_attributes[row["track_id"]]
            row["brand"] = label.brand

    if metadata_rows:
        db.bulk_insert_mappings(EvaluationMetadata, metadata_rows)
        db.commit()

    return len(metadata_rows)


def load_labels(labels_json_path: str | Path | None) -> Labels:
    if labels_json_path is None:
        return Labels(yolo={}, classifier={})

    with Path(labels_json_path).open("r", encoding="utf-8") as labels_file:
        data = json.load(labels_file)

    yolo_mapping = _mapping_from_keys(
        data,
        keys=("yolo", "yolo_classes", "detector", "vehicle_types", "vehicle_type"),
    )
    classifier_mapping = _mapping_from_keys(
        data,
        keys=("classifier", "classification", "classifier_classes", "brand_color", "labels"),
    )

    return Labels(
        yolo=parse_yolo_labels(yolo_mapping),
        classifier=parse_classifier_labels(classifier_mapping),
    )


def load_roi_polygon(roi_json_path: str | Path | None, cctv_id: str) -> list[Point] | None:
    if roi_json_path is None:
        return None

    path = Path(roi_json_path)
    if not path.is_file():
        raise FileNotFoundError(f"ROI JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as roi_file:
        data = json.load(roi_file)

    raw_polygon = data.get(cctv_id) if isinstance(data, dict) else None
    if raw_polygon is None:
        return None

    polygon: list[Point] = []
    for point in raw_polygon:
        if isinstance(point, dict):
            polygon.append((float(point["x"]), float(point["y"])))
        else:
            x, y = point
            polygon.append((float(x), float(y)))

    if len(polygon) < 3:
        raise ValueError(f"ROI for {cctv_id} must contain at least 3 points")
    return polygon


def point_in_polygon(x: float, y: float, polygon: list[Point]) -> bool:
    inside = False
    previous_x, previous_y = polygon[-1]

    for current_x, current_y in polygon:
        crosses_y = (current_y > y) != (previous_y > y)
        if crosses_y:
            slope_x = (previous_x - current_x) * (y - current_y)
            slope_x /= previous_y - current_y
            intersection_x = slope_x + current_x
            if x < intersection_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y

    return inside


def _mapping_from_keys(data: Any, keys: tuple[str, ...]) -> Any:
    if not isinstance(data, dict):
        return data
    for key in keys:
        if key in data:
            return data[key]
    return data


def parse_yolo_labels(raw: Any) -> dict[int, str]:
    if isinstance(raw, list):
        return {index: str(label) for index, label in enumerate(raw)}
    if isinstance(raw, dict):
        labels: dict[int, str] = {}
        for key, value in raw.items():
            try:
                class_id = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                value = value.get("name") or value.get("label") or value.get("vehicle_type")
            labels[class_id] = str(value)
        return labels
    return {}


def parse_classifier_labels(raw: Any) -> dict[int, ClassificationLabel]:
    if isinstance(raw, list):
        return {index: parse_classifier_label(label) for index, label in enumerate(raw)}
    if isinstance(raw, dict):
        labels: dict[int, ClassificationLabel] = {}
        for key, value in raw.items():
            try:
                class_id = int(key)
            except (TypeError, ValueError):
                continue
            labels[class_id] = parse_classifier_label(value)
        return labels
    return {}


def parse_classifier_label(value: Any) -> ClassificationLabel:
    if isinstance(value, dict):
        return ClassificationLabel(
            brand=_clean_label(
                value.get("brand")
                or value.get("make")
                or value.get("name")
                or value.get("label")
            ),
        )

    text = _clean_label(value)
    if text is None:
        return ClassificationLabel(brand=None)

    for separator in ("_", ",", "|", "/"):
        if separator in text:
            brand, _ignored = text.split(separator, 1)
            return ClassificationLabel(brand=_clean_label(brand))

    return ClassificationLabel(brand=text)


def _clean_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().strip("\"'")
    return text or None


def resolve_yolo_label(class_id: int, labels: Labels, model_names: dict[int, str] | list[str]) -> str:
    if class_id in labels.yolo:
        return labels.yolo[class_id]
    if isinstance(model_names, dict) and class_id in model_names:
        return str(model_names[class_id])
    if isinstance(model_names, list) and class_id < len(model_names):
        return str(model_names[class_id])
    return DEFAULT_YOLO_NAMES.get(class_id, str(class_id))


def normalize_vehicle_type(value: str) -> str:
    text = str(value).strip().lower()
    aliases = {
        "auto": "car",
        "automobile": "car",
        "vehicle": "car",
        "motorbike": "motorcycle",
        "motor cycle": "motorcycle",
        "lorry": "truck",
    }
    return aliases.get(text, text)


def crop_frame(frame: Any, x1: float, y1: float, x2: float, y2: float) -> Any | None:
    height, width = frame.shape[:2]
    left = max(0, min(width - 1, int(round(x1))))
    top = max(0, min(height - 1, int(round(y1))))
    right = max(0, min(width, int(round(x2))))
    bottom = max(0, min(height, int(round(y2))))
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right]


def classify_vehicle(classifier_session: Any, crop: Any, labels: Labels) -> ClassificationLabel:
    cv2 = _import_cv2()
    np = _import_numpy()

    resized = cv2.resize(crop, CLASSIFIER_IMAGE_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = (tensor - mean) / std
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

    input_name = classifier_session.get_inputs()[0].name
    outputs = classifier_session.run(None, {input_name: tensor})
    logits = outputs[0]
    class_id = int(np.asarray(logits).reshape(-1).argmax())
    return labels.classifier.get(class_id, ClassificationLabel(brand=str(class_id)))


def _to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def _load_yolo(yolo_model_path: str | Path) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required to run YOLO ONNX tracking") from exc
    return YOLO(str(yolo_model_path), task="detect")


def _load_classifier_session(classifier_model_path: str | Path) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required to run classifier.onnx") from exc
    return ort.InferenceSession(str(classifier_model_path), providers=["CPUExecutionProvider"])


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required to process videos") from exc
    return cv2


def _import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for classifier preprocessing") from exc
    return np
