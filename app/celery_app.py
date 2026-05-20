import os

from celery import Celery


celery_app = Celery(
    "leaderboard_worker",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
    include=["app.worker"],
)

celery_app.conf.update(
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_time_limit=int(os.getenv("CELERY_TASK_TIME_LIMIT", "7200")),
    task_soft_time_limit=int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "7000")),
)
