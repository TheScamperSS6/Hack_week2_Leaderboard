# Docker Compose Deployment

This deployment runs the full app on one central machine:

- Next.js frontend on port `3000`
- FastAPI backend on port `8000`
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
INTERNAL_API_BASE_URL=http://backend:8000
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
http://YOUR_LAN_IP:8000/health/db
```

The leaderboard defaults to Type Mode ranked by ACC:

```text
http://YOUR_LAN_IP:8000/leaderboard
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
