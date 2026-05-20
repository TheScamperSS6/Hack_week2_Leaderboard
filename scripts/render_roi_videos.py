import argparse
import json
from pathlib import Path

import cv2


ROI_COLOR = (0, 255, 255)
POINT_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", default="data/videos.json")
    parser.add_argument("--rois", default="data/rois.json")
    parser.add_argument("--out-dir", default="storage/roi_videos")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    args = parser.parse_args()

    videos = json.loads(Path(args.videos).read_text(encoding="utf-8"))
    rois = json.loads(Path(args.rois).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cctv_id, video_path in videos.items():
        output_path = out_dir / f"{cctv_id}_roi.mp4"
        render_video(
            cctv_id=cctv_id,
            video_path=Path(video_path),
            roi=rois.get(cctv_id, []),
            output_path=output_path,
            scale=args.scale,
            max_seconds=args.max_seconds,
        )
        print(output_path)


def render_video(
    cctv_id: str,
    video_path: Path,
    roi: list[list[float]],
    output_path: Path,
    scale: float,
    max_seconds: float,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_limit = int(fps * max_seconds) if max_seconds > 0 else None
    output_size = (int(width * scale), int(height * scale))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        output_size,
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"failed to create output video: {output_path}")

    scaled_roi = [(int(x * scale), int(y * scale)) for x, y in roi]
    frame_index = 0

    try:
        while True:
            if frame_limit is not None and frame_index >= frame_limit:
                break

            ok, frame = capture.read()
            if not ok:
                break

            if scale != 1.0:
                frame = cv2.resize(frame, output_size)

            draw_roi(frame, cctv_id, scaled_roi)
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()


def draw_roi(frame, cctv_id: str, points: list[tuple[int, int]]) -> None:
    cv2.putText(
        frame,
        f"{cctv_id} ROI",
        (24, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        TEXT_COLOR,
        3,
        cv2.LINE_AA,
    )

    if len(points) < 3:
        return

    for start, end in zip(points, points[1:] + points[:1], strict=False):
        cv2.line(frame, start, end, ROI_COLOR, 3)

    for index, point in enumerate(points, start=1):
        cv2.circle(frame, point, 6, POINT_COLOR, -1)
        cv2.putText(
            frame,
            str(index),
            (point[0] + 7, point[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            POINT_COLOR,
            2,
            cv2.LINE_AA,
        )


if __name__ == "__main__":
    main()
