from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models import SubmissionStatus


class SubmissionCreated(BaseModel):
    id: int
    user_id: int
    team_name: str | None = None
    status: SubmissionStatus
    evaluation_mode: str
    yolo_model_path: str
    class_model_path: str | None
    labels_json_path: str | None
    description: str | None
    yolo_gflops: float
    class_gflops: float
    evaluation_task_id: str | None = None

    model_config = ConfigDict(from_attributes=True)


class LeaderboardEntry(BaseModel):
    submission_id: int
    user_id: int
    team_name: str
    status: SubmissionStatus
    evaluation_mode: str
    description: str | None
    acc_score: float | None
    eff_score: float | None
    yolo_gflops: float
    class_gflops: float
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SubmissionDetail(BaseModel):
    submission_id: int
    user_id: int
    team_name: str
    status: SubmissionStatus
    evaluation_mode: str
    description: str | None
    acc_score: float | None
    eff_score: float | None
    yolo_gflops: float
    class_gflops: float
    yolo_model_path: str | None
    class_model_path: str | None
    labels_json_path: str | None
    metadata_count: int
    created_at: datetime


class QuestionResult(BaseModel):
    question_id: str
    cctv_id: str
    time_range: str
    query: str
    group_by: list[str]
    prediction: dict[str, int]
    ground_truth: dict[str, int]
    acc_score: float


class PreviewVideo(BaseModel):
    cctv_id: str
    video_url: str
    file_path: str
    exists: bool


class SubmissionResults(BaseModel):
    submission_id: int
    user_id: int
    team_name: str
    status: SubmissionStatus
    evaluation_mode: str
    description: str | None
    acc_score: float | None
    eff_score: float | None
    metadata_count: int
    questions_csv_path: str
    answers_csv_path: str
    question_results: list[QuestionResult]
    preview_videos: list[PreviewVideo]


class PreviewGenerationResult(BaseModel):
    submission_id: int
    preview_videos: list[PreviewVideo]


class SubmissionScore(BaseModel):
    submission_id: int
    acc_score: float
    eff_score: float


class ProcessPendingRequest(BaseModel):
    evaluation_mode: str | None = None
    cctv_id: str | None = None
    video_path: str | None = None
    videos: list["EvaluationVideo"] | None = None
    questions_csv_path: str
    answers_csv_path: str
    roi_json_path: str | None = None


class EvaluationVideo(BaseModel):
    cctv_id: str
    video_path: str


class TaskCreated(BaseModel):
    task_id: str
    status: str


class TaskStatus(BaseModel):
    task_id: str
    status: str
    result: object | None = None
    error: str | None = None
