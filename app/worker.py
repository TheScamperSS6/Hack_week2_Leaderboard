import traceback
from pathlib import Path

from celery.utils.log import get_task_logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.celery_app import celery_app
from app.database import SessionLocal
from app.evaluation import score_submission_from_csv
from app.models import Submission, SubmissionStatus
from app.vision_pipeline import run_video_pipeline


logger = get_task_logger(__name__)


class SubmissionCancelled(Exception):
    pass


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

    queued_task_ids: dict[int, str] = {}
    for submission_id in submission_ids:
        task = process_submission.delay(
            submission_id=submission_id,
            video_jobs=video_jobs,
            questions_csv_path=questions_csv_path,
            answers_csv_path=answers_csv_path,
            roi_json_path=roi_json_path,
        )
        queued_task_ids[submission_id] = task.id

    if queued_task_ids:
        with SessionLocal() as db:
            submissions = db.scalars(
                select(Submission).where(Submission.id.in_(queued_task_ids))
            )
            for submission in submissions:
                submission.evaluation_task_id = queued_task_ids[submission.id]
            db.commit()

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
) -> dict[str, float | int | str]:
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
        if submission.status == SubmissionStatus.cancelled:
            logger.info("Skipping cancelled submission %s", submission_id)
            return {
                "submission_id": submission_id,
                "metadata_rows": 0,
                "status": "cancelled",
            }
        if submission.status != SubmissionStatus.pending:
            logger.info("Skipping submission %s with status %s", submission_id, submission.status)
            return {"submission_id": submission_id, "metadata_rows": 0}

        submission.status = SubmissionStatus.processing
        submission.error_message = None
        db.commit()

        try:
            _ensure_file_exists(submission.yolo_model_path)
            if submission.class_model_path is not None:
                _ensure_file_exists(submission.class_model_path)
            if submission.labels_json_path is not None:
                _ensure_file_exists(submission.labels_json_path)

            metadata_rows = 0
            for video_job in normalized_video_jobs:
                raise_if_submission_cancelled(db, submission.id)
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
                    should_cancel=lambda: raise_if_submission_cancelled(db, submission.id),
                )
            raise_if_submission_cancelled(db, submission.id)
            total_acc, eff_score = score_submission_from_csv(
                db=db,
                submission=submission,
                questions_csv_path=questions_csv_path,
                answers_csv_path=answers_csv_path,
            )
            submission.acc_score = total_acc
            submission.eff_score = eff_score
            submission.status = SubmissionStatus.done
            submission.error_message = None
            db.commit()
        except SubmissionCancelled as exc:
            db.rollback()
            submission = db.get(Submission, submission_id)
            if submission is not None:
                submission.status = SubmissionStatus.cancelled
                submission.error_message = str(exc)
                submission.acc_score = None
                submission.eff_score = None
                db.commit()
            logger.info("Cancelled submission %s", submission_id)
            return {
                "submission_id": submission_id,
                "metadata_rows": 0,
                "status": "cancelled",
            }
        except Exception as exc:
            error_message = format_failure_message(exc)
            db.rollback()
            submission = db.get(Submission, submission_id)
            if submission is not None:
                submission.status = SubmissionStatus.failed
                submission.error_message = error_message
                submission.acc_score = None
                submission.eff_score = None
                db.commit()
            logger.exception("Failed to process submission %s", submission_id)
            return {
                "submission_id": submission_id,
                "metadata_rows": 0,
                "status": "failed",
                "error_message": error_message,
            }

    return {
        "submission_id": submission_id,
        "metadata_rows": metadata_rows,
        "acc_score": total_acc,
        "eff_score": eff_score,
    }


def raise_if_submission_cancelled(db: Session, submission_id: int) -> None:
    current_status = db.scalar(
        select(Submission.status).where(Submission.id == submission_id)
    )
    if current_status == SubmissionStatus.cancelled:
        raise SubmissionCancelled("Cancelled by admin")


def format_failure_message(exc: Exception) -> str:
    message = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return message[-8000:]


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
