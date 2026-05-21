import json
import os
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from celery.result import AsyncResult
from sqlalchemy import desc, func, inspect, nullslast, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.celery_app import celery_app
from app.database import Base, engine, get_db
from app.evaluation import question_results_from_csv, score_submission_from_csv
from app.model_stats import estimate_onnx_gflops
from app.preview import preview_video_infos, render_detection_previews
from app.worker import process_pending_submissions, process_submission
from app.models import EvaluationMetadata, Submission, SubmissionStatus, User
from app.schemas import (
    LeaderboardEntry,
    PreviewGenerationResult,
    ProcessPendingRequest,
    SubmissionCreated,
    SubmissionDetail,
    SubmissionResults,
    SubmissionScore,
    TaskCreated,
    TaskStatus,
)


STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", "storage/submissions"))
PREVIEW_ROOT = Path(os.getenv("PREVIEW_ROOT", "storage/previews"))
PREVIEW_SCALE = float(os.getenv("PREVIEW_SCALE", "0.75"))
AUTO_PROCESS_SUBMISSIONS = os.getenv("AUTO_PROCESS_SUBMISSIONS", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
AUTO_PROCESS_PAYLOADS = {
    "brand": Path(os.getenv("AUTO_PROCESS_BRAND_PAYLOAD", "data/process_payload_brand.example.json")),
    "type": Path(os.getenv("AUTO_PROCESS_TYPE_PAYLOAD", "data/process_payload_type.example.json")),
}
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]


def _run_startup_migrations() -> None:
    """Small compatibility migration for local dev DBs created before Alembic."""
    with engine.begin() as connection:
        inspector = inspect(connection)
        table_names = inspector.get_table_names()
        if "submissions" in table_names:
            submission_columns = {
                column["name"]
                for column in inspector.get_columns("submissions")
            }
            if "evaluation_mode" not in submission_columns:
                connection.execute(
                    text(
                        "ALTER TABLE submissions "
                        "ADD COLUMN evaluation_mode VARCHAR(20) NOT NULL DEFAULT 'brand'"
                    )
                )
                connection.execute(
                    text(
                        "UPDATE submissions "
                        "SET evaluation_mode = 'type' "
                        "WHERE class_model_path IS NULL"
                    )
                )
            if "description" not in submission_columns:
                connection.execute(
                    text("ALTER TABLE submissions ADD COLUMN description TEXT")
                )

        if "evaluation_metadata" not in table_names:
            return

        columns = {
            column["name"]
            for column in inspector.get_columns("evaluation_metadata")
        }
        if "cctv_id" not in columns:
            connection.execute(
                text(
                    "ALTER TABLE evaluation_metadata "
                    "ADD COLUMN cctv_id VARCHAR(80) NOT NULL DEFAULT 'CCTV01'"
                )
            )
        if "color" not in columns:
            connection.execute(
                text("ALTER TABLE evaluation_metadata ADD COLUMN color VARCHAR(120)")
            )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    Base.metadata.create_all(bind=engine)
    _run_startup_migrations()
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    yield


PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)
app = FastAPI(
    title="Leaderboard Evaluation Platform",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount(
    "/storage/previews",
    StaticFiles(directory=PREVIEW_ROOT),
    name="preview_videos",
)


@app.get("/health/db")
def check_database() -> dict[str, str]:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"database connection failed: {exc}",
        ) from exc

    return {"status": "ok"}


def _validate_extension(upload: UploadFile, expected_suffix: str) -> None:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix != expected_suffix:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{upload.filename or 'file'} must be a {expected_suffix} file",
        )


def _save_upload(upload: UploadFile, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)


def _validate_labels_json(labels_file: UploadFile) -> None:
    try:
        labels_file.file.seek(0)
        json.load(labels_file.file)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="labels.json is not valid JSON",
        ) from exc
    finally:
        labels_file.file.seek(0)


def _get_or_create_user(
    db: Session,
    user_id: int | None,
    team_name: str | None,
) -> User:
    if user_id is not None:
        user = db.get(User, user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"user_id {user_id} was not found",
            )
        return user

    cleaned_team_name = (team_name or "").strip()
    if not cleaned_team_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="team_name is required when user_id is not provided",
        )

    existing_user = db.scalar(select(User).where(User.team_name == cleaned_team_name))
    if existing_user is not None:
        return existing_user

    user = User(team_name=cleaned_team_name)
    db.add(user)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        user = db.scalar(select(User).where(User.team_name == cleaned_team_name))
        if user is None:
            raise
    return user


def _load_process_payload(evaluation_mode: str) -> dict:
    payload_path = AUTO_PROCESS_PAYLOADS[evaluation_mode]
    if not payload_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"auto-process payload file was not found: {payload_path}",
        )

    with payload_path.open("r", encoding="utf-8") as payload_file:
        return json.load(payload_file)


def _enqueue_submission_evaluation(submission_id: int, evaluation_mode: str) -> str | None:
    if not AUTO_PROCESS_SUBMISSIONS:
        return None

    payload = _load_process_payload(evaluation_mode)
    task = process_submission.delay(
        submission_id=submission_id,
        questions_csv_path=payload["questions_csv_path"],
        answers_csv_path=payload["answers_csv_path"],
        cctv_id=payload.get("cctv_id"),
        video_path=payload.get("video_path"),
        video_jobs=payload.get("videos"),
        roi_json_path=payload.get("roi_json_path"),
    )
    return task.id


def _video_jobs_from_payload(payload: dict) -> list[dict[str, str]]:
    videos = payload.get("videos")
    if videos:
        return [
            {"cctv_id": str(video["cctv_id"]), "video_path": str(video["video_path"])}
            for video in videos
        ]

    cctv_id = payload.get("cctv_id")
    video_path = payload.get("video_path")
    if cctv_id and video_path:
        return [{"cctv_id": str(cctv_id), "video_path": str(video_path)}]

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="process payload must include either videos[] or cctv_id + video_path",
    )


def _resolve_gflops(
    model_name: str,
    provided_gflops: float | None,
    model_path: Path,
) -> float:
    if provided_gflops is not None:
        if provided_gflops <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{model_name}_gflops must be a positive number when provided",
            )
        return provided_gflops

    try:
        return estimate_onnx_gflops(model_path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"could not estimate {model_name} GFLOPs from ONNX. "
                "Please enter GFLOPs manually."
            ),
        ) from exc


@app.post(
    "/submit",
    response_model=SubmissionCreated,
    status_code=status.HTTP_201_CREATED,
)
def submit_models(
    user_id: int | None = Form(None),
    team_name: str | None = Form(None),
    description: str | None = Form(None),
    evaluation_mode: str = Form("brand"),
    yolo_gflops: float | None = Form(None),
    class_gflops: float | None = Form(None),
    yolo_model: UploadFile = File(..., description="YOLO ONNX file"),
    class_model: UploadFile | None = File(None, description="Classifier ONNX file"),
    labels_json: UploadFile | None = File(None, description="Labels JSON file"),
    db: Session = Depends(get_db),
) -> Submission:
    user = _get_or_create_user(db, user_id, team_name)
    normalized_mode = evaluation_mode.strip().lower()
    if normalized_mode not in {"brand", "type"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="evaluation_mode must be either 'brand' or 'type'",
        )

    if normalized_mode == "brand":
        if class_model is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="classifier.onnx is required in brand mode",
            )
        if labels_json is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="labels.json is required in brand mode",
            )

    _validate_extension(yolo_model, ".onnx")
    if class_model is not None:
        _validate_extension(class_model, ".onnx")
    if labels_json is not None:
        _validate_extension(labels_json, ".json")
        _validate_labels_json(labels_json)

    submission = Submission(
        user_id=user.id,
        status=SubmissionStatus.pending,
        evaluation_mode=normalized_mode,
        description=(description or "").strip() or None,
        yolo_gflops=0.0,
        class_gflops=0.0,
    )
    db.add(submission)
    db.flush()

    submission_dir = STORAGE_ROOT / str(submission.id)
    yolo_path = submission_dir / "yolo.onnx"
    class_path = submission_dir / "classifier.onnx" if class_model is not None else None
    labels_path = submission_dir / "labels.json" if labels_json is not None else None

    try:
        _save_upload(yolo_model, yolo_path)
        if class_model is not None and class_path is not None:
            _save_upload(class_model, class_path)
        if labels_json is not None and labels_path is not None:
            _save_upload(labels_json, labels_path)
        submission.yolo_gflops = _resolve_gflops("yolo", yolo_gflops, yolo_path)
        submission.class_gflops = (
            _resolve_gflops("class", class_gflops, class_path)
            if normalized_mode == "brand" and class_path is not None
            else 0.0
        )
    except HTTPException:
        db.rollback()
        if submission_dir.exists():
            shutil.rmtree(submission_dir, ignore_errors=True)
        raise
    except OSError as exc:
        db.rollback()
        if submission_dir.exists():
            shutil.rmtree(submission_dir, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="failed to save uploaded files",
        ) from exc

    submission.yolo_model_path = str(yolo_path)
    submission.class_model_path = str(class_path) if class_path is not None else None
    submission.labels_json_path = str(labels_path) if labels_path is not None else None
    db.commit()
    db.refresh(submission)
    submission.team_name = user.team_name
    submission.evaluation_task_id = _enqueue_submission_evaluation(
        submission_id=submission.id,
        evaluation_mode=normalized_mode,
    )
    return submission


@app.get("/leaderboard", response_model=list[LeaderboardEntry])
def get_leaderboard(
    mode: str = Query("type", pattern="^(type|brand)$"),
    sort_by: str = Query("acc", pattern="^(eff|acc)$"),
    db: Session = Depends(get_db),
) -> list[LeaderboardEntry]:
    sort_column = Submission.acc_score if sort_by == "acc" else Submission.eff_score
    rows = db.execute(
        select(
            Submission.id.label("submission_id"),
            Submission.user_id,
            User.team_name,
            Submission.status,
            Submission.evaluation_mode,
            Submission.description,
            Submission.acc_score,
            Submission.eff_score,
            Submission.yolo_gflops,
            Submission.class_gflops,
            Submission.created_at,
        )
        .join(User, User.id == Submission.user_id)
        .where(Submission.evaluation_mode == mode)
        .order_by(nullslast(desc(sort_column)), Submission.created_at.asc())
    ).mappings()

    return [LeaderboardEntry.model_validate(row) for row in rows]


@app.get("/submissions/{submission_id}", response_model=SubmissionDetail)
def get_submission_status(
    submission_id: int,
    db: Session = Depends(get_db),
) -> SubmissionDetail:
    row = db.execute(
        select(
            Submission.id.label("submission_id"),
            Submission.user_id,
            User.team_name,
            Submission.status,
            Submission.evaluation_mode,
            Submission.description,
            Submission.acc_score,
            Submission.eff_score,
            Submission.yolo_gflops,
            Submission.class_gflops,
            Submission.yolo_model_path,
            Submission.class_model_path,
            Submission.labels_json_path,
            func.count(EvaluationMetadata.id).label("metadata_count"),
            Submission.created_at,
        )
        .join(User, User.id == Submission.user_id)
        .outerjoin(EvaluationMetadata, EvaluationMetadata.submission_id == Submission.id)
        .where(Submission.id == submission_id)
        .group_by(Submission.id, User.id)
    ).mappings().one_or_none()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"submission_id {submission_id} was not found",
        )

    return SubmissionDetail.model_validate(row)


@app.get("/submissions/{submission_id}/results", response_model=SubmissionResults)
def get_submission_results(
    submission_id: int,
    db: Session = Depends(get_db),
) -> SubmissionResults:
    row = db.execute(
        select(
            Submission.id.label("submission_id"),
            Submission.user_id,
            User.team_name,
            Submission.status,
            Submission.evaluation_mode,
            Submission.description,
            Submission.acc_score,
            Submission.eff_score,
            func.count(EvaluationMetadata.id).label("metadata_count"),
        )
        .join(User, User.id == Submission.user_id)
        .outerjoin(EvaluationMetadata, EvaluationMetadata.submission_id == Submission.id)
        .where(Submission.id == submission_id)
        .group_by(Submission.id, User.id)
    ).mappings().one_or_none()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"submission_id {submission_id} was not found",
        )

    payload = _load_process_payload(str(row["evaluation_mode"]))
    questions_csv_path = payload["questions_csv_path"]
    answers_csv_path = payload["answers_csv_path"]
    video_jobs = _video_jobs_from_payload(payload)
    question_results = question_results_from_csv(
        db=db,
        submission_id=submission_id,
        questions_csv_path=questions_csv_path,
        answers_csv_path=answers_csv_path,
    )

    return SubmissionResults(
        **row,
        questions_csv_path=questions_csv_path,
        answers_csv_path=answers_csv_path,
        question_results=question_results,
        preview_videos=preview_video_infos(submission_id, video_jobs, PREVIEW_ROOT),
    )


@app.post("/submissions/{submission_id}/previews", response_model=PreviewGenerationResult)
def generate_submission_previews(
    submission_id: int,
    db: Session = Depends(get_db),
) -> PreviewGenerationResult:
    submission = db.get(Submission, submission_id)
    if submission is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"submission_id {submission_id} was not found",
        )

    metadata_count = db.scalar(
        select(func.count(EvaluationMetadata.id)).where(
            EvaluationMetadata.submission_id == submission_id
        )
    )
    if not metadata_count:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="no evaluation metadata is available for this submission yet",
        )

    payload = _load_process_payload(submission.evaluation_mode)
    video_jobs = _video_jobs_from_payload(payload)
    previews = render_detection_previews(
        db=db,
        submission_id=submission_id,
        video_jobs=video_jobs,
        preview_root=PREVIEW_ROOT,
        roi_json_path=payload.get("roi_json_path"),
        scale=PREVIEW_SCALE,
    )
    return PreviewGenerationResult(
        submission_id=submission_id,
        preview_videos=previews,
    )


@app.post("/submissions/{submission_id}/rescore", response_model=SubmissionScore)
def rescore_submission(
    submission_id: int,
    db: Session = Depends(get_db),
) -> SubmissionScore:
    submission = db.get(Submission, submission_id)
    if submission is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"submission_id {submission_id} was not found",
        )

    metadata_count = db.scalar(
        select(func.count(EvaluationMetadata.id)).where(
            EvaluationMetadata.submission_id == submission_id
        )
    )
    if not metadata_count:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="no evaluation metadata is available for this submission yet",
        )

    payload = _load_process_payload(submission.evaluation_mode)
    total_acc, eff_score = score_submission_from_csv(
        db=db,
        submission=submission,
        questions_csv_path=payload["questions_csv_path"],
        answers_csv_path=payload["answers_csv_path"],
    )
    submission.acc_score = total_acc
    submission.eff_score = eff_score
    db.commit()

    return SubmissionScore(
        submission_id=submission_id,
        acc_score=total_acc,
        eff_score=eff_score,
    )


@app.post("/process-pending", response_model=TaskCreated)
def enqueue_pending_submissions(payload: ProcessPendingRequest) -> TaskCreated:
    task = process_pending_submissions.delay(
        evaluation_mode=payload.evaluation_mode,
        cctv_id=payload.cctv_id,
        video_path=payload.video_path,
        videos=[video.model_dump() for video in payload.videos] if payload.videos else None,
        questions_csv_path=payload.questions_csv_path,
        answers_csv_path=payload.answers_csv_path,
        roi_json_path=payload.roi_json_path,
    )
    return TaskCreated(task_id=task.id, status=task.status)


@app.get("/tasks/{task_id}", response_model=TaskStatus)
def get_task_status(task_id: str) -> TaskStatus:
    task = AsyncResult(task_id, app=celery_app)
    error = str(task.result) if task.failed() else None
    result = task.result if task.successful() else None
    return TaskStatus(
        task_id=task.id,
        status=task.status,
        result=result,
        error=error,
    )
