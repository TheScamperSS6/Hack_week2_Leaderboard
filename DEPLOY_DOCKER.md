# Docker Compose Deployment

This deployment runs the full app on one central machine:

- Next.js frontend on port `3000`
- FastAPI backend on port `8001`
- Celery worker
- Redis
- PostgreSQL is expected to run on the host machine, outside Docker

## 1. Prepare Files

```bash
git pull
cp .env.deploy.example .env
```

Edit `.env`:

```env
DATABASE_URL=postgresql+psycopg2://postgres:YOUR_PASSWORD@host.docker.internal:5432/leaderboard
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://YOUR_LAN_IP:3000
NEXT_PUBLIC_API_BASE_URL=/api
INTERNAL_API_BASE_URL=http://backend:8001
NEXT_PUBLIC_UPLOAD_API_BASE_URL=
NEXT_PUBLIC_BACKEND_PORT=8001
```

Put the evaluation videos on the host:

```text
data/videos/clip_000.mp4
data/videos/clip_001.mp4
data/videos/clip_002.mp4
data/videos/clip_003.mp4
data/videos/clip_004.mp4
```

These video files are intended to be committed to git for this LAN deployment.
They are still excluded from the Docker build context by `.dockerignore`, because
Docker Compose bind-mounts `./data` into the backend and worker containers at runtime.

## 2. PostgreSQL Host Access

The containers connect to PostgreSQL through:

```text
host.docker.internal
```

On Linux Docker, PostgreSQL usually must listen beyond `127.0.0.1`.
Check `postgresql.conf`:

```conf
listen_addresses = '*'
```

And allow the Docker bridge network in `pg_hba.conf`, for example:

```conf
host    leaderboard    postgres    172.16.0.0/12    md5
```

Then restart PostgreSQL.

## 3. Start

```bash
docker compose up -d --build
```

## Dev Mode Without Rebuilding

For active code edits, run Compose with the dev override:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
```

After that first build, code changes are bind-mounted from the host:

- FastAPI reloads automatically.
- Next.js runs in dev mode and hot-reloads UI changes.
- Celery worker restarts when Python files change.

For normal code-only updates:

```bash
git pull
docker compose -f docker-compose.yml -f docker-compose.dev.yml restart worker
```

Rebuild only when dependencies or Dockerfiles change, such as
`requirements.txt`, `package.json`, `package-lock.json`, `Dockerfile.backend`,
or `Dockerfile.frontend`.

If the frontend feels slow because Next.js keeps compiling in dev mode, use the
backend-dev override instead. It keeps FastAPI and Celery live-mounted, but runs
the frontend in production mode:

```bash
docker compose -f docker-compose.yml -f docker-compose.backend-dev.yml up -d --build
```

For code-only backend updates with that mode:

```bash
git pull
docker compose -f docker-compose.yml -f docker-compose.backend-dev.yml restart backend worker
```

Use this mode for demos or LAN usage when the UI is already stable.

After dependency changes, rebuild the backend image so both `backend` and `worker`
get the new Python packages:

```bash
docker compose build --no-cache backend worker
docker compose up -d
```

Open from another machine on the same Wi-Fi:

```text
http://YOUR_LAN_IP:3000
```

Backend health check:

```text
http://YOUR_LAN_IP:8001/health/db
```

The leaderboard defaults to Type Mode ranked by ACC:

```text
http://YOUR_LAN_IP:8001/leaderboard
```

## CPU Performance Tuning

The worker defaults to:

```env
EVAL_TARGET_FPS=10
YOLO_IMGSZ=416
YOLO_CONF=0.25
TRACKER_BACKEND=auto
TRACKER_MAX_FRAME_ERRORS=25
SIMPLE_TRACKER_IOU=0.30
SIMPLE_TRACKER_MAX_AGE=30
```

For CPU-only machines, this default is the recommended starting point:

```env
EVAL_TARGET_FPS=5
YOLO_IMGSZ=416
YOLO_CONF=0.25
TRACKER_BACKEND=auto
TRACKER_MAX_FRAME_ERRORS=25
SIMPLE_TRACKER_IOU=0.30
SIMPLE_TRACKER_MAX_AGE=30
```

Faster but riskier:

```env
EVAL_TARGET_FPS=3
YOLO_IMGSZ=320
YOLO_CONF=0.30
TRACKER_BACKEND=auto
TRACKER_MAX_FRAME_ERRORS=25
SIMPLE_TRACKER_IOU=0.30
SIMPLE_TRACKER_MAX_AGE=30
```

After changing these values in `.env`, restart the worker:

```bash
docker compose restart worker
```

`TRACKER_BACKEND=auto` uses ByteTrack first and falls back to a simple IoU
tracker if ByteTrack hits numerical errors. Use `TRACKER_BACKEND=simple` to skip
ByteTrack entirely for models that consistently break Ultralytics tracking.

## Upload Size

Model upload goes directly from the browser to FastAPI on port `8001`, bypassing
the Next.js `/api` proxy so large ONNX and `.onnx.data` files can submit reliably.
Make sure client machines can reach the backend port and `ALLOWED_ORIGINS`
contains the frontend origin, for example `http://YOUR_LAN_IP:3000`.

Leave this empty to auto-use the same hostname as the frontend:

```env
NEXT_PUBLIC_UPLOAD_API_BASE_URL=
NEXT_PUBLIC_BACKEND_PORT=8001
```

If the backend is exposed through another URL, set it explicitly:

```env
NEXT_PUBLIC_UPLOAD_API_BASE_URL=http://YOUR_LAN_IP:8001
```

Some exported ONNX files use external data, for example
`vehiclemakenet.onnx.data`. In Brand Mode, upload that sidecar file in the
`classifier external data` field with its original filename.

The `/api` proxy limit is also set to `200mb` as a fallback. If frontend env vars
or `next.config.mjs` change, rebuild the frontend image:

```bash
docker compose build frontend
docker compose up -d frontend
```

## 4. Logs

```bash
docker compose logs -f backend
docker compose logs -f worker
docker compose logs -f frontend
```

## 5. Offline Machine Note

If the central machine cannot reach Docker Hub, PyPI, npm, or apt repositories,
build the images on a machine with internet first:

```bash
docker compose build
docker save cv-leaderboard-backend:latest cv-leaderboard-frontend:latest redis:7-alpine -o cv-leaderboard-images.tar
```

Move `cv-leaderboard-images.tar` to the central machine:

```bash
docker load -i cv-leaderboard-images.tar
docker compose up -d
```
