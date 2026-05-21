import json
import logging
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.models import EvaluationMetadata


TARGET_FPS = float(os.getenv("EVAL_TARGET_FPS", "10.0"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "416"))
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.25"))
TRACKER_BACKEND = os.getenv("TRACKER_BACKEND", "auto").strip().lower()
TRACKER_MAX_FRAME_ERRORS = int(os.getenv("TRACKER_MAX_FRAME_ERRORS", "25"))
SIMPLE_TRACKER_IOU = float(os.getenv("SIMPLE_TRACKER_IOU", "0.30"))
SIMPLE_TRACKER_MAX_AGE = int(os.getenv("SIMPLE_TRACKER_MAX_AGE", "30"))
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

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassificationLabel:
    brand: str | None


@dataclass(frozen=True)
class Labels:
    yolo: dict[int, str]
    classifier: dict[int, ClassificationLabel]


Point = tuple[float, float]
RoiMap = dict[str, list[Point]]
BBox = tuple[float, float, float, float]


@dataclass
class SimpleTrack:
    track_id: int
    class_id: int
    bbox: BBox
    missing_frames: int = 0


class SimpleIouTracker:
    def __init__(
        self,
        iou_threshold: float = SIMPLE_TRACKER_IOU,
        max_age: int = SIMPLE_TRACKER_MAX_AGE,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_track_id = 1
        self.tracks: dict[int, SimpleTrack] = {}

    def update(self, boxes: list[BBox], class_ids: list[int]) -> list[int]:
        for track in self.tracks.values():
            track.missing_frames += 1

        assigned_track_ids: list[int] = []
        used_tracks: set[int] = set()
        for bbox, class_id in zip(boxes, class_ids, strict=False):
            best_track: SimpleTrack | None = None
            best_iou = 0.0
            for track in self.tracks.values():
                if track.track_id in used_tracks or track.class_id != class_id:
                    continue
                score = bbox_iou(bbox, track.bbox)
                if score > best_iou:
                    best_iou = score
                    best_track = track

            if best_track is not None and best_iou >= self.iou_threshold:
                best_track.bbox = bbox
                best_track.missing_frames = 0
                assigned_track_ids.append(best_track.track_id)
                used_tracks.add(best_track.track_id)
                continue

            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = SimpleTrack(
                track_id=track_id,
                class_id=class_id,
                bbox=bbox,
            )
            assigned_track_ids.append(track_id)
            used_tracks.add(track_id)

        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if track.missing_frames <= self.max_age
        }
        return assigned_track_ids


class RawYoloOnnxDetector:
    def __init__(
        self,
        session: Any,
        input_name: str,
        output_name: str,
        imgsz: int | list[int],
        class_count: int,
    ) -> None:
        self.session = session
        self.input_name = input_name
        self.output_name = output_name
        self.imgsz = imgsz
        self.class_count = class_count
        self.names = default_raw_yolo_names(class_count)

    def predict(self, frame: Any) -> tuple[list[BBox], list[int], dict[int, str]]:
        np = _import_numpy()
        image, ratio, pad_x, pad_y, target_width, target_height = letterbox_frame(
            frame,
            self.imgsz,
        )
        image = image[:, :, ::-1].transpose(2, 0, 1)
        tensor = np.ascontiguousarray(image, dtype=np.float32)[None] / 255.0

        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        predictions = np.asarray(outputs[0])
        if predictions.ndim == 3:
            predictions = predictions[0]
        if predictions.ndim != 2:
            return [], [], self.names

        expected_columns = self.class_count + 5
        if predictions.shape[1] != expected_columns and predictions.shape[0] == expected_columns:
            predictions = predictions.T
        if predictions.shape[1] != expected_columns:
            return [], [], self.names

        boxes_xywh = predictions[:, :4].astype(np.float32)
        objectness = predictions[:, 4].astype(np.float32)
        class_scores = predictions[:, 5:].astype(np.float32)
        class_ids = class_scores.argmax(axis=1).astype(np.int32)
        scores = objectness * class_scores.max(axis=1)
        keep = scores >= YOLO_CONF
        if not bool(keep.any()):
            return [], [], self.names

        boxes_xywh = boxes_xywh[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        if float(boxes_xywh.max(initial=0.0)) <= 2.0:
            boxes_xywh[:, [0, 2]] *= target_width
            boxes_xywh[:, [1, 3]] *= target_height

        boxes_xyxy = yolo_xywh_to_original_xyxy(
            boxes_xywh,
            ratio=ratio,
            pad_x=pad_x,
            pad_y=pad_y,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
        )
        keep_indices = classwise_nms(boxes_xyxy, scores, class_ids)

        boxes: list[BBox] = []
        kept_class_ids: list[int] = []
        for index in keep_indices:
            bbox = tuple(float(value) for value in boxes_xyxy[index])
            if not is_valid_bbox(bbox):
                continue
            boxes.append(bbox)
            kept_class_ids.append(int(class_ids[index]))

        return boxes, kept_class_ids, self.names


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
    raw_detector = create_raw_yolo_onnx_detector(yolo_model_path)
    if raw_detector is None:
        yolo_model = _load_yolo(yolo_model_path)
        yolo_imgsz = resolve_yolo_imgsz(yolo_model_path)
    else:
        logger.info(
            "Using raw YOLO ONNX parser for %s with imgsz=%s and %s classes",
            yolo_model_path,
            raw_detector.imgsz,
            raw_detector.class_count,
        )
        yolo_model = None
        yolo_imgsz = raw_detector.imgsz
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
    tracker_frame_errors = 0
    track_id_offset = 0
    simple_tracker = SimpleIouTracker()
    use_simple_tracker = raw_detector is not None or TRACKER_BACKEND == "simple"

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

            if raw_detector is not None:
                xyxy_rows, class_ids, names = raw_detector.predict(frame)
                if not xyxy_rows:
                    continue
                track_ids = simple_tracker.update(xyxy_rows, class_ids)
            elif use_simple_tracker:
                if yolo_model is None:
                    raise RuntimeError("YOLO model is not loaded")
                try:
                    result = yolo_model.predict(
                        frame,
                        device="cpu",
                        imgsz=yolo_imgsz,
                        conf=YOLO_CONF,
                        verbose=False,
                    )[0]
                except Exception as exc:
                    expected_imgsz = expected_imgsz_from_onnx_error(exc)
                    if expected_imgsz is None:
                        raise
                    logger.warning(
                        "ONNX input size mismatch for submission=%s cctv_id=%s. "
                        "Switching YOLO imgsz from %s to %s.",
                        submission_id,
                        cctv_id,
                        yolo_imgsz,
                        expected_imgsz,
                    )
                    yolo_imgsz = expected_imgsz
                    yolo_model = _load_yolo(yolo_model_path)
                    continue

                xyxy_rows, class_ids, _result_track_ids = detections_from_result(result)
                if not xyxy_rows:
                    continue
                track_ids = simple_tracker.update(xyxy_rows, class_ids)
                names = getattr(result, "names", None) or getattr(yolo_model, "names", {})
            else:
                if yolo_model is None:
                    raise RuntimeError("YOLO model is not loaded")
                try:
                    result = yolo_model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        device="cpu",
                        imgsz=yolo_imgsz,
                        conf=YOLO_CONF,
                        verbose=False,
                    )[0]
                except Exception as exc:
                    expected_imgsz = expected_imgsz_from_onnx_error(exc)
                    if expected_imgsz is not None:
                        logger.warning(
                            "ONNX input size mismatch for submission=%s cctv_id=%s. "
                            "Switching YOLO imgsz from %s to %s.",
                            submission_id,
                            cctv_id,
                            yolo_imgsz,
                            expected_imgsz,
                        )
                        yolo_imgsz = expected_imgsz
                        yolo_model = _load_yolo(yolo_model_path)
                        continue

                    if not is_recoverable_tracker_error(exc):
                        raise

                    tracker_frame_errors += 1
                    if TRACKER_BACKEND == "bytetrack":
                        if tracker_frame_errors > TRACKER_MAX_FRAME_ERRORS:
                            raise RuntimeError(
                                "YOLO tracker failed repeatedly with numerical errors. "
                                f"Last frame={frame_index}, timestamp={timestamp:.2f}s"
                            ) from exc

                        logger.warning(
                            "Resetting ByteTrack after numerical tracker error "
                            "for submission=%s cctv_id=%s frame=%s timestamp=%.2fs: %s",
                            submission_id,
                            cctv_id,
                            frame_index,
                            timestamp,
                            exc,
                        )
                        track_id_offset += 100000
                        yolo_model = _load_yolo(yolo_model_path)
                        continue

                    logger.warning(
                        "ByteTrack failed for submission=%s cctv_id=%s frame=%s "
                        "timestamp=%.2fs; falling back to simple IoU tracker: %s",
                        submission_id,
                        cctv_id,
                        frame_index,
                        timestamp,
                        exc,
                    )
                    yolo_model = _load_yolo(yolo_model_path)
                    simple_tracker.next_track_id = max(
                        simple_tracker.next_track_id,
                        1000000,
                    )
                    use_simple_tracker = True
                    try:
                        result = yolo_model.predict(
                            frame,
                            device="cpu",
                            imgsz=yolo_imgsz,
                            conf=YOLO_CONF,
                            verbose=False,
                        )[0]
                    except Exception as predict_exc:
                        expected_imgsz = expected_imgsz_from_onnx_error(predict_exc)
                        if expected_imgsz is None:
                            raise
                        logger.warning(
                            "ONNX input size mismatch for submission=%s cctv_id=%s. "
                            "Switching YOLO imgsz from %s to %s.",
                            submission_id,
                            cctv_id,
                            yolo_imgsz,
                            expected_imgsz,
                        )
                        yolo_imgsz = expected_imgsz
                        yolo_model = _load_yolo(yolo_model_path)
                        continue

                    xyxy_rows, class_ids, _result_track_ids = detections_from_result(result)
                    if not xyxy_rows:
                        continue
                    track_ids = simple_tracker.update(xyxy_rows, class_ids)
                    names = getattr(result, "names", None) or getattr(yolo_model, "names", {})
                else:
                    xyxy_rows, class_ids, result_track_ids = detections_from_result(result)
                    if not xyxy_rows or result_track_ids is None:
                        continue
                    track_ids = [
                        int(track_id) + track_id_offset
                        for track_id in result_track_ids
                    ]
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


def create_raw_yolo_onnx_detector(yolo_model_path: str | Path) -> RawYoloOnnxDetector | None:
    path = Path(yolo_model_path)
    if path.suffix.lower() != ".onnx":
        return None

    try:
        import onnxruntime as ort
    except ImportError:
        return None

    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]
    except Exception:
        logger.debug("Unable to inspect raw YOLO ONNX output for %s", path, exc_info=True)
        return None

    output_shape = output_meta.shape
    if len(output_shape) != 3:
        return None

    anchor_dim, column_dim = output_shape[1], output_shape[2]
    if not isinstance(anchor_dim, int) or not isinstance(column_dim, int):
        return None
    if anchor_dim <= column_dim or column_dim < 7:
        return None
    raw_columns = column_dim

    class_count = raw_columns - 5
    # Raw YOLOv5 exports look like [1, anchors, 5 + classes]. Already postprocessed
    # detector outputs usually have 6 columns, so leave those to Ultralytics.
    if class_count <= 1:
        return None

    input_imgsz = image_size_from_input_shape(input_meta.shape) or YOLO_IMGSZ
    return RawYoloOnnxDetector(
        session=session,
        input_name=input_meta.name,
        output_name=output_meta.name,
        imgsz=input_imgsz,
        class_count=class_count,
    )


def image_size_from_input_shape(shape: list[Any]) -> int | list[int] | None:
    if len(shape) < 4:
        return None
    height, width = shape[2], shape[3]
    if not isinstance(height, int) or not isinstance(width, int):
        return None
    if height <= 0 or width <= 0:
        return None
    if height == width:
        return height
    return [height, width]


def default_raw_yolo_names(class_count: int) -> dict[int, str]:
    if class_count == 4:
        return {
            0: "car",
            1: "truck",
            2: "bus",
            3: "motorcycle",
        }
    return {class_id: f"class{class_id}" for class_id in range(class_count)}


def letterbox_frame(
    frame: Any,
    imgsz: int | list[int],
) -> tuple[Any, float, int, int, int, int]:
    cv2 = _import_cv2()
    if isinstance(imgsz, list):
        target_height, target_width = int(imgsz[0]), int(imgsz[1])
    else:
        target_height = target_width = int(imgsz)

    frame_height, frame_width = frame.shape[:2]
    ratio = min(target_width / frame_width, target_height / frame_height)
    resized_width = int(round(frame_width * ratio))
    resized_height = int(round(frame_height * ratio))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    pad_width = target_width - resized_width
    pad_height = target_height - resized_height
    left = int(round(pad_width / 2 - 0.1))
    right = int(round(pad_width / 2 + 0.1))
    top = int(round(pad_height / 2 - 0.1))
    bottom = int(round(pad_height / 2 + 0.1))
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, ratio, left, top, target_width, target_height


def yolo_xywh_to_original_xyxy(
    boxes_xywh: Any,
    ratio: float,
    pad_x: int,
    pad_y: int,
    frame_width: int,
    frame_height: int,
) -> Any:
    np = _import_numpy()
    boxes = boxes_xywh.astype(np.float32).copy()
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / ratio
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / ratio
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, frame_width)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, frame_height)
    return xyxy


def classwise_nms(boxes: Any, scores: Any, class_ids: Any, iou_threshold: float = 0.45) -> list[int]:
    np = _import_numpy()
    keep: list[int] = []
    unique_classes = sorted(int(class_id) for class_id in set(class_ids.tolist()))
    for class_id in unique_classes:
        indices = np.where(class_ids == class_id)[0]
        indices = indices[np.argsort(scores[indices])[::-1]]
        while len(indices) > 0:
            current = int(indices[0])
            keep.append(current)
            if len(indices) == 1:
                break

            rest = indices[1:]
            ious = np.array(
                [bbox_iou(tuple(boxes[current]), tuple(boxes[index])) for index in rest],
                dtype=np.float32,
            )
            indices = rest[ious <= iou_threshold]

    keep.sort(key=lambda index: float(scores[index]), reverse=True)
    return keep[:300]


def detections_from_result(result: Any) -> tuple[list[BBox], list[int], list[int] | None]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return [], [], None

    xyxy_rows = _to_numpy(boxes.xyxy)
    class_ids = _to_numpy(boxes.cls).astype(int)
    raw_track_ids = None
    if getattr(boxes, "id", None) is not None:
        raw_track_ids = _to_numpy(boxes.id).astype(int)

    detections: list[BBox] = []
    detection_class_ids: list[int] = []
    detection_track_ids: list[int] | None = [] if raw_track_ids is not None else None

    for index, (xyxy, class_id) in enumerate(zip(xyxy_rows, class_ids, strict=False)):
        bbox = tuple(float(value) for value in xyxy)
        if not is_valid_bbox(bbox):
            continue

        detections.append(bbox)
        detection_class_ids.append(int(class_id))
        if detection_track_ids is not None and raw_track_ids is not None:
            detection_track_ids.append(int(raw_track_ids[index]))

    return detections, detection_class_ids, detection_track_ids


def is_valid_bbox(bbox: BBox) -> bool:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return (
        all(math.isfinite(value) for value in bbox)
        and width >= 2.0
        and height >= 2.0
    )


def bbox_iou(first: BBox, second: BBox) -> float:
    first_x1, first_y1, first_x2, first_y2 = first
    second_x1, second_y1, second_x2, second_y2 = second

    inter_x1 = max(first_x1, second_x1)
    inter_y1 = max(first_y1, second_y1)
    inter_x2 = min(first_x2, second_x2)
    inter_y2 = min(first_y2, second_y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    first_area = max(0.0, first_x2 - first_x1) * max(0.0, first_y2 - first_y1)
    second_area = max(0.0, second_x2 - second_x1) * max(0.0, second_y2 - second_y1)
    union = first_area + second_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


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


def is_recoverable_tracker_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    if exc.__class__.__name__ == "LinAlgError":
        return True
    return (
        "not positive definite" in message
        or "singular matrix" in message
        or "cholesky" in message
    )


def expected_imgsz_from_onnx_error(exc: Exception) -> int | None:
    message = str(exc)
    if "invalid dimensions" not in message.casefold():
        return None

    expected_values = [
        int(value)
        for value in re.findall(r"Expected:\s*(\d+)", message)
        if int(value) > 0
    ]
    if not expected_values:
        return None

    first_value = expected_values[0]
    if any(value != first_value for value in expected_values):
        return None
    return first_value


def resolve_yolo_imgsz(yolo_model_path: str | Path) -> int | list[int]:
    fixed_imgsz = fixed_onnx_input_imgsz(str(yolo_model_path))
    if fixed_imgsz is not None:
        if fixed_imgsz != YOLO_IMGSZ:
            logger.info(
                "Using fixed YOLO ONNX input size %s instead of YOLO_IMGSZ=%s",
                fixed_imgsz,
                YOLO_IMGSZ,
            )
        return fixed_imgsz
    return YOLO_IMGSZ


@lru_cache(maxsize=64)
def fixed_onnx_input_imgsz(yolo_model_path: str) -> int | list[int] | None:
    path = Path(yolo_model_path)
    if path.suffix.lower() != ".onnx":
        return None

    try:
        import onnxruntime as ort
    except ImportError:
        return None

    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        shape = session.get_inputs()[0].shape
    except Exception:
        logger.debug("Unable to inspect ONNX input size for %s", path, exc_info=True)
        return None

    if len(shape) < 4:
        return None

    height, width = shape[2], shape[3]
    if not isinstance(height, int) or not isinstance(width, int):
        return None
    if height <= 0 or width <= 0:
        return None

    if height == width:
        return height
    return [height, width]


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
