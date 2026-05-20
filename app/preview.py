import re
import shutil
import subprocess
import time
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import EvaluationMetadata
from app.vision_pipeline import TARGET_FPS, load_roi_polygon


TYPE_COLORS = {
    "car": (255, 190, 40),
    "truck": (35, 145, 255),
    "bus": (185, 90, 255),
    "motorcycle": (80, 210, 120),
}
DEFAULT_COLOR = (240, 240, 240)
TEXT_COLOR = (255, 255, 255)
HEADER_BG = (15, 23, 42)
ROI_COLOR = (0, 255, 255)
ROI_FILL_COLOR = (0, 180, 255)


def preview_video_infos(
    submission_id: int,
    video_jobs: list[dict[str, str]],
    preview_root: Path,
) -> list[dict[str, object]]:
    return [
        preview_video_info(submission_id, video_job["cctv_id"], preview_root)
        for video_job in video_jobs
    ]


def preview_video_info(
    submission_id: int,
    cctv_id: str,
    preview_root: Path,
) -> dict[str, object]:
    output_path = latest_preview_video_path(submission_id, cctv_id, preview_root)
    return {
        "cctv_id": cctv_id,
        "video_url": f"/storage/previews/{submission_id}/{output_path.name}",
        "file_path": str(output_path),
        "exists": output_path.is_file(),
    }


def render_detection_previews(
    db: Session,
    submission_id: int,
    video_jobs: list[dict[str, str]],
    preview_root: Path,
    roi_json_path: str | Path | None = None,
    scale: float = 0.75,
) -> list[dict[str, object]]:
    rendered: list[dict[str, object]] = []
    version = int(time.time() * 1000)
    for video_job in video_jobs:
        cctv_id = video_job["cctv_id"]
        output_path = versioned_preview_video_path(
            submission_id,
            cctv_id,
            preview_root,
            version,
        )
        render_detection_preview(
            db=db,
            submission_id=submission_id,
            cctv_id=cctv_id,
            video_path=Path(video_job["video_path"]),
            output_path=output_path,
            roi_json_path=roi_json_path,
            scale=scale,
        )
        rendered.append(preview_video_info_for_path(cctv_id, output_path, submission_id))
    return rendered


def render_detection_preview(
    db: Session,
    submission_id: int,
    cctv_id: str,
    video_path: Path,
    output_path: Path,
    roi_json_path: str | Path | None = None,
    scale: float = 0.75,
) -> None:
    cv2 = _import_cv2()
    metadata_by_timestamp = _metadata_by_timestamp(db, submission_id, cctv_id)
    roi_polygon = load_roi_polygon(roi_json_path, cctv_id)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open input video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_output_path = output_path.with_name(f"{output_path.stem}.raw.mp4")
    writer = cv2.VideoWriter(
        str(temp_output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        TARGET_FPS,
        output_size,
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"failed to create preview video: {output_path}")

    frame_index = -1
    next_sample_time = 0.0
    completed = False

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

            if scale != 1.0:
                frame = cv2.resize(frame, output_size)

            rows = metadata_by_timestamp.get(_timestamp_key(timestamp), [])
            draw_roi_overlay(frame, roi_polygon, scale)
            draw_header(frame, cctv_id, timestamp, len(rows))
            for row in rows:
                draw_detection(frame, row, scale)
            writer.write(frame)
        completed = True
    finally:
        capture.release()
        writer.release()

    if completed:
        convert_to_browser_mp4(temp_output_path, output_path)


def _metadata_by_timestamp(
    db: Session,
    submission_id: int,
    cctv_id: str,
) -> dict[float, list[EvaluationMetadata]]:
    rows = db.scalars(
        select(EvaluationMetadata)
        .where(EvaluationMetadata.submission_id == submission_id)
        .where(EvaluationMetadata.cctv_id == cctv_id)
        .order_by(EvaluationMetadata.timestamp.asc(), EvaluationMetadata.track_id.asc())
    )

    grouped: dict[float, list[EvaluationMetadata]] = {}
    for row in rows:
        grouped.setdefault(_timestamp_key(row.timestamp), []).append(row)
    return grouped


def draw_header(frame, cctv_id: str, timestamp: float, count: int) -> None:
    cv2 = _import_cv2()
    label = f"{cctv_id}  {timestamp:06.2f}s  detections: {count}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), HEADER_BG, -1)
    cv2.putText(
        frame,
        label,
        (14, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )


def draw_roi_overlay(frame, roi_polygon: list[tuple[float, float]] | None, scale: float) -> None:
    if not roi_polygon:
        return

    cv2 = _import_cv2()
    np = _import_numpy()
    points = np.array(
        [[int(x * scale), int(y * scale)] for x, y in roi_polygon],
        dtype=np.int32,
    )
    if len(points) < 3:
        return

    overlay = frame.copy()
    cv2.fillPoly(overlay, [points], ROI_FILL_COLOR)
    cv2.addWeighted(overlay, 0.14, frame, 0.86, 0, dst=frame)
    cv2.polylines(frame, [points], isClosed=True, color=ROI_COLOR, thickness=3)

    label_x = int(points[:, 0].min())
    label_y = max(42, int(points[:, 1].min()) - 8)
    cv2.putText(
        frame,
        "ROI",
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        ROI_COLOR,
        3,
        cv2.LINE_AA,
    )


def draw_detection(frame, row: EvaluationMetadata, scale: float) -> None:
    cv2 = _import_cv2()
    x1 = int(row.bbox_x * scale)
    y1 = int(row.bbox_y * scale)
    x2 = int((row.bbox_x + row.bbox_w) * scale)
    y2 = int((row.bbox_y + row.bbox_h) * scale)
    color = TYPE_COLORS.get(row.vehicle_type, DEFAULT_COLOR)
    label_name = row.brand or row.vehicle_type
    label = f"#{row.track_id} {label_name}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (label_width, label_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        2,
    )
    label_top = max(36, y1 - label_height - baseline - 6)
    cv2.rectangle(
        frame,
        (x1, label_top),
        (x1 + label_width + 8, label_top + label_height + baseline + 6),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 4, label_top + label_height + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (15, 23, 42),
        2,
        cv2.LINE_AA,
    )


def preview_video_info_for_path(
    cctv_id: str,
    output_path: Path,
    submission_id: int,
) -> dict[str, object]:
    return {
        "cctv_id": cctv_id,
        "video_url": f"/storage/previews/{submission_id}/{output_path.name}",
        "file_path": str(output_path),
        "exists": output_path.is_file(),
    }


def latest_preview_video_path(submission_id: int, cctv_id: str, preview_root: Path) -> Path:
    base = safe_filename(cctv_id)
    preview_dir = preview_root / str(submission_id)
    candidates = [
        *preview_dir.glob(f"{base}_*.mp4"),
        preview_dir / f"{base}.mp4",
    ]
    existing_candidates = [path for path in candidates if path.is_file()]
    if not existing_candidates:
        return preview_dir / f"{base}.mp4"
    return max(existing_candidates, key=lambda path: path.stat().st_mtime_ns)


def versioned_preview_video_path(
    submission_id: int,
    cctv_id: str,
    preview_root: Path,
    version: int,
) -> Path:
    return preview_root / str(submission_id) / f"{safe_filename(cctv_id)}_{version}.mp4"


def convert_to_browser_mp4(source_path: Path, output_path: Path) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        source_path.replace(output_path)
        return

    command = [
        ffmpeg_path,
        "-y",
        "-v",
        "error",
        "-i",
        str(source_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True)
    finally:
        source_path.unlink(missing_ok=True)


def safe_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return text or "preview"


def _timestamp_key(timestamp: float) -> float:
    return round(float(timestamp), 3)


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required to render preview videos") from exc
    return cv2


def _import_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to render preview videos") from exc
    return np
