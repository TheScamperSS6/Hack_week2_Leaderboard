from pathlib import Path

from celery.utils.log import get_task_logger
from sqlalchemy import select

from app.celery_app import celery_app
from app.database import SessionLocal
from app.evaluation import score_submission_from_csv
from app.models import Submission, SubmissionStatus
from app.vision_pipeline import run_video_pipeline


logger = get_task_logger(__name__)


@celery_app.task(name="process_pending_submissions")
def process_pending_submissions(
    questions_csv_path: str,
    answers_csv_path: str,
    evaluation_mode: str | None = None,
    cctv_id: str | None = None,
    video_path: str | None = None,
    roi_json_path: str | None = None,
    videos: list[dict[str, str]] | None = None,
) -> list[int]:
    video_jobs = normalize_video_jobs(cctv_id, video_path, videos)
    for video_job in video_jobs:
        _ensure_file_exists(video_job["video_path"])

    with SessionLocal() as db:
        statement = (
            select(Submission.id)
            .where(Submission.status == SubmissionStatus.pending)
            .order_by(Submission.created_at.asc())
        )
        if evaluation_mode:
            statement = statement.where(
                Submission.evaluation_mode == evaluation_mode.strip().lower()
            )
        submission_ids = list(
            db.scalars(statement)
        )

    for submission_id in submission_ids:
        process_submission.delay(
            submission_id=submission_id,
            video_jobs=video_jobs,
            questions_csv_path=questions_csv_path,
            answers_csv_path=answers_csv_path,
            roi_json_path=roi_json_path,
        )

    return submission_ids


@celery_app.task(name="process_submission", bind=True)
def process_submission(
    self,
    submission_id: int,
    questions_csv_path: str,
    answers_csv_path: str,
    cctv_id: str | None = None,
    video_path: str | None = None,
    roi_json_path: str | None = None,
    video_jobs: list[dict[str, str]] | None = None,
) -> dict[str, float | int]:
    normalized_video_jobs = normalize_video_jobs(cctv_id, video_path, video_jobs)
    for video_job in normalized_video_jobs:
        _ensure_file_exists(video_job["video_path"])
    _ensure_file_exists(questions_csv_path)
    _ensure_file_exists(answers_csv_path)
    if roi_json_path is not None:
        _ensure_file_exists(roi_json_path)

    with SessionLocal() as db:
        submission = db.get(Submission, submission_id)
        if submission is None:
            raise ValueError(f"submission_id {submission_id} was not found")
        if submission.status != SubmissionStatus.pending:
            logger.info("Skipping submission %s with status %s", submission_id, submission.status)
            return {"submission_id": submission_id, "metadata_rows": 0}

        _ensure_file_exists(submission.yolo_model_path)
        if submission.class_model_path is not None:
            _ensure_file_exists(submission.class_model_path)
        if submission.labels_json_path is not None:
            _ensure_file_exists(submission.labels_json_path)

        submission.status = SubmissionStatus.processing
        db.commit()

        try:
            metadata_rows = 0
            for video_job in normalized_video_jobs:
                metadata_rows += run_video_pipeline(
                    db=db,
                    submission_id=submission.id,
                    cctv_id=video_job["cctv_id"],
                    video_path=video_job["video_path"],
                    yolo_model_path=submission.yolo_model_path,
                    classifier_model_path=(
                        submission.class_model_path
                        if submission.evaluation_mode == "brand"
                        else None
                    ),
                    labels_json_path=submission.labels_json_path,
                    roi_json_path=roi_json_path,
                )
            total_acc, eff_score = score_submission_from_csv(
                db=db,
                submission=submission,
                questions_csv_path=questions_csv_path,
                answers_csv_path=answers_csv_path,
            )
            submission.acc_score = total_acc
            submission.eff_score = eff_score
            submission.status = SubmissionStatus.done
            db.commit()
        except Exception:
            db.rollback()
            submission = db.get(Submission, submission_id)
            if submission is not None:
                submission.status = SubmissionStatus.pending
                db.commit()
            logger.exception("Failed to process submission %s", submission_id)
            raise

    return {
        "submission_id": submission_id,
        "metadata_rows": metadata_rows,
        "acc_score": total_acc,
        "eff_score": eff_score,
    }


def _ensure_file_exists(path: str | Path | None) -> None:
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(f"file not found: {path}")


def normalize_video_jobs(
    cctv_id: str | None,
    video_path: str | None,
    videos: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    if videos:
        return [
            {"cctv_id": str(video["cctv_id"]), "video_path": str(video["video_path"])}
            for video in videos
        ]

    if cctv_id and video_path:
        return [{"cctv_id": cctv_id, "video_path": video_path}]

    raise ValueError("provide either cctv_id + video_path or videos[]")
