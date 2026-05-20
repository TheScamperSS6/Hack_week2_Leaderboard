import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", default="data/videos.json")
    parser.add_argument("--rois", default="data/rois.json")
    parser.add_argument("--out-dir", default="storage/roi_previews")
    parser.add_argument("--at-seconds", type=float, default=30.0)
    parser.add_argument("--grid-step", type=int, default=100)
    args = parser.parse_args()

    videos = json.loads(Path(args.videos).read_text(encoding="utf-8"))
    rois = json.loads(Path(args.rois).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cctv_id, video_path in videos.items():
        frame = read_frame(Path(video_path), args.at_seconds)
        polygon = rois.get(cctv_id)
        if polygon:
            points = [(int(x), int(y)) for x, y in polygon]
            draw_grid(frame, args.grid_step)
            for start, end in zip(points, points[1:] + points[:1], strict=False):
                cv2.line(frame, start, end, (0, 255, 255), 4)
            for index, point in enumerate(points, start=1):
                cv2.circle(frame, point, 8, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    str(index),
                    (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        cv2.putText(
            frame,
            cctv_id,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(out_dir / f"{cctv_id}.jpg"), frame)
        print(out_dir / f"{cctv_id}.jpg")


def read_frame(video_path: Path, at_seconds: float):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(at_seconds * fps))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"failed to read frame from: {video_path}")
    return frame


def draw_grid(frame, step: int) -> None:
    if step <= 0:
        return

    height, width = frame.shape[:2]
    for x in range(0, width, step):
        cv2.line(frame, (x, 0), (x, height), (255, 255, 255), 1)
        cv2.putText(
            frame,
            str(x),
            (x + 4, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    for y in range(0, height, step):
        cv2.line(frame, (0, y), (width, y), (255, 255, 255), 1)
        cv2.putText(
            frame,
            str(y),
            (8, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


if __name__ == "__main__":
    main()
