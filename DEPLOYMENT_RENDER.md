# Migrating from AWS App Runner to Render

This guide covers the full migration of the Energy Demand ML API from AWS App Runner to Render: the new `render.yaml`, the Dockerfile changes that make the image Render-compatible, model-artifact handling under Render's ephemeral free-tier filesystem, and an external cron job that stops the free instance from spinning down.

Files changed/added in this repo as part of this migration: `render.yaml` (new), `Dockerfile`, `.dockerignore`, `docker-compose.yml`.

## 1. Why App Runner's setup doesn't work as-is on Render

| Concern | App Runner | Render |
|---|---|---|
| Port binding | Fixed port (80), configured in the App Runner service | Render injects a dynamic `PORT` env var (typically `10000`); the container must bind to it |
| Health checks | Configured in the App Runner console, hits the container's fixed port | Configured via `healthCheckPath` in `render.yaml`, hits whatever port `$PORT` resolves to |
| Idle behavior | Runs continuously (no free tier) | Free plan spins down after 15 minutes with no inbound traffic |
| Filesystem | Ephemeral, but service never sleeps mid-request | Ephemeral, and *also* wiped on every spin-down/restart/redeploy |
| Deploy trigger | GitHub Actions builds and pushes an image to ECR | Render builds directly from your Dockerfile on git push (Blueprint-based) |

The one blocking incompatibility is the port: your current Dockerfile hardcodes `--port 80` and `EXPOSE 80`. Render assigns its own port via `$PORT` — if the app doesn't bind to it, health checks fail and the service never comes up. That's the first fix below.

## 2. Dockerfile changes

Already applied to `Dockerfile`:

- Added `ENV PORT=80` as a **default fallback** (used by local `docker run`/`docker-compose`, where nothing sets `PORT`).
- Changed `CMD` to `sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"` (exec-array `CMD` can't expand env vars, so it's now shell form).
- Changed `HEALTHCHECK` to use `curl` against `${PORT}` instead of hardcoded port 80 (also switched from a Python one-liner to `curl`, which is lighter and doesn't depend on the `requests` package still being importable).
- Installed `curl` in the `apt-get` layer.

Render **ignores** `EXPOSE` and the Docker `HEALTHCHECK` instruction for routing/health decisions — it uses `healthCheckPath` from `render.yaml` against the `$PORT` it assigned. Both are kept for local Docker/Compose use and for portability to other orchestrators.

No other Dockerfile changes are required. `pip install -r requirements.txt` and the `mkdir -p` step for `/app/models`, `/app/data`, `/app/app/static` behave the same on Render as they did on App Runner.

## 3. Model artifacts — the part that actually needs a decision

`models/*.pkl`, `data/processed/*.csv`, and `data/raw/*.csv`/`*.zip` are all in `.gitignore`. That's fine on a machine where you trained the model locally and then built+pushed the image (the files were present in your build context even though git didn't track them). It becomes a real problem on Render, because:

- Render builds your image straight from what's in the **git repo** — gitignored files never reach the build.
- Render's free plan has **no persistent disk**, and *any* local filesystem write (including a model produced by hitting `POST /train` after deploy) is wiped on the next spin-down, restart, or redeploy.

Without addressing this, `app/main.py`'s `startup_event()` will run, find no `models/xgboost_demand_model.pkl`, and `/health` will report `"status": "unhealthy"` forever. Pick one:

**Option A — commit the trained artifacts (recommended for the free tier).**
Train locally as you already do, then remove the two `models/*.pkl` / `models/*.joblib` lines from `.gitignore` and commit `models/`. XGBoost + encoder + metrics pickles for this feature set are typically a few MB — trivial for git. This is the simplest option and requires no build-time training.

```bash
git add -f models/xgboost_demand_model.pkl models/feature_names.pkl models/label_encoders.pkl models/model_metrics.pkl
```

**Option B — train the model as part of the Docker build.**
Add a `RUN python src/etl.py && RUN python src/train.py` step before the final `CMD`. This requires `data/raw/*.csv` (currently gitignored) to also be committed or fetched during the build, and adds real time to every Render build. Only worth it if you don't want binary model files in git history.

**Option C — fetch the model from external storage at container startup.**
Store the `.pkl` files in S3/R2/GCS and download them in an entrypoint script before `uvicorn` starts. Most flexible, but adds a dependency and startup latency, and is overkill unless the model is large or retrained frequently outside the deploy pipeline.

If you go with A or B, `data/raw/*.json` (already committed) plus whatever raw data `etl.py` needs should stay in the repo; the `.dockerignore` update below has commented-out lines for excluding `data/raw`/`data/processed` from the build context — only uncomment those if you're on Option A (they're not needed at runtime, just don't remove them if Option B still needs them at build time).

## 4. `.dockerignore` and `docker-compose.yml`

`.dockerignore` now also excludes `images/`, `README.md`, `docker-compose.yml`, `.github/`, and `*.ipynb` from the build context — none of these are needed inside the running container, and trimming them speeds up Render's build (Render's free plan also has monthly build-pipeline-minute limits).

`docker-compose.yml` now sets `PORT=80` explicitly and its healthcheck uses `curl` to match the Dockerfile, so local `docker-compose up` behaves identically to before.

## 5. `render.yaml`

```yaml
services:
  - type: web
    name: energy-demand-ml-api
    runtime: docker
    dockerfilePath: ./Dockerfile
    dockerContext: .
    plan: free
    region: oregon
    branch: main
    autoDeploy: true
    healthCheckPath: /health
    envVars:
      - key: PYTHONUNBUFFERED
        value: "1"
```

This is a Render **Blueprint**: point Render at a repo containing `render.yaml` and it provisions/updates services to match. Key fields:

- `runtime: docker` + `dockerfilePath` / `dockerContext` — build from your existing Dockerfile, no buildpack guessing.
- `plan: free` — 512 MB RAM / 0.5 CPU, 750 free instance-hours/month, spins down after 15 min idle. Change to `starter` ($7/mo) later to remove spin-down entirely if this ever needs to behave like a real production service.
- `healthCheckPath: /health` — Render polls this after each deploy to decide whether the new instance is healthy before routing traffic to it (zero-downtime deploys), and reports the result in the dashboard.
- `autoDeploy: true` — every push to `main` triggers a new build+deploy, replacing the GitHub Actions → ECR → App Runner pipeline. You do **not** need `.github/workflows/deploy.yml` anymore for this purpose (see §7).
- `region: oregon` — change if your users/data sources are elsewhere (`ohio`, `virginia`, `frankfurt`, `singapore` are the other options).

## 6. Step-by-step deployment

1. Decide on and implement a model-artifact strategy from §3 (Option A is the fast path). Commit `render.yaml`, the updated `Dockerfile`, `.dockerignore`, and `docker-compose.yml`, and push to `main`.
2. Sign in at [dashboard.render.com](https://dashboard.render.com) (create an account if needed) and connect your GitHub account/repo.
3. Click **New → Blueprint**, select this repository. Render detects `render.yaml` and shows a preview of the `energy-demand-ml-api` web service it's about to create.
4. Click **Apply**. Render clones the repo, builds the Docker image from `Dockerfile`, and starts the container. Watch the build logs in the dashboard — first build will take a few minutes (installing `pandas`/`xgboost`/`scikit-learn` from scratch).
5. Once deployed, Render assigns a URL like `https://energy-demand-ml-api.onrender.com`. Verify:
   - `GET /health` → `{"status": "healthy", "model_loaded": true, ...}` (if this says `unhealthy`/`model_loaded: false`, go back to §3 — the model artifacts didn't make it into the image)
   - `GET /dashboard` → renders the dashboard page
   - `GET /docs` → Swagger UI
   - `POST /predict/xgboost` with the example payload from the README → returns a prediction
6. (Optional) Add a custom domain under the service's **Settings → Custom Domains** if you were using one on App Runner.
7. Set up the external cron ping (§8) before considering the migration complete — otherwise the service will sleep after 15 minutes of no traffic and every first request after that eats a ~30–60s cold start.

## 7. Retiring the AWS pipeline

Once the Render deployment is verified:

- Delete or disable `.github/workflows/deploy.yml` (or repurpose it as a test/lint workflow — strip the ECR login/push steps, keep it for CI checks on PRs if useful).
- Remove `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` from the repo's GitHub Actions secrets.
- In the AWS Console: delete the App Runner service, and either delete the `energy-predict-api` ECR repository or keep it briefly as a rollback fallback before removing it.
- Update `README.md`'s "Deployed with FastAPI and AWS" line and any deployment instructions to reference Render instead (not required functionally, but avoids confusing future contributors).

## 8. External cron job to prevent spin-down

Render's free web services spin down after **15 minutes with no inbound traffic** (HTTP requests or WebSocket messages), and take about a minute to spin back up on the next request. `healthCheckPath` in `render.yaml` only affects Render's own deploy-health checks — it does **not** ping your service on a schedule, so it does nothing to prevent idle spin-down. You need something outside Render hitting `/health` periodically.

**Important trade-off:** the free plan grants **750 instance-hours/workspace/month**. A 31-day month has 744 hours, so keeping this one service alive around the clock via cron pings will consume essentially your entire monthly free allowance, leaving no headroom for other free services in the same Render workspace. If that's a problem, either only ping during the hours you need it responsive, or move to the `starter` plan ($7/mo), which removes spin-down and the instance-hour cap.

### Recommended: cron-job.org (free, 1-minute granularity, no card required)

1. Create a free account at [cron-job.org](https://cron-job.org).
2. Click **Create cronjob**.
3. **Title:** `energy-demand-ml-api keepalive`
4. **URL:** `https://<your-service>.onrender.com/health`
5. **Schedule:** every 10 minutes (comfortably under the 15-minute spin-down window; leaves margin if a single execution is delayed).
6. **Request method:** `GET`.
7. Save. cron-job.org shows execution history and HTTP status codes for each run — a `200` confirms the ping kept the service awake; anything else (or a request that reads slow the first time) means the service had already spun down and this ping cold-started it.

### Alternative: UptimeRobot (adds uptime monitoring + alerting)

Free plan checks every 5 minutes and can email/Slack you on downtime, at the cost of less granular scheduling than cron-job.org:

1. Sign up at [uptimerobot.com](https://uptimerobot.com).
2. **Add New Monitor** → type **HTTP(s)**.
3. **URL:** `https://<your-service>.onrender.com/health`.
4. **Monitoring interval:** 5 minutes (fixed on the free plan).
5. Optionally add an alert contact (email) so you're notified if `/health` ever returns a non-2xx status, not just idle spin-down.

Either service works — UptimeRobot is a better fit if you also want failure alerting; cron-job.org is a better fit if you specifically want the ping frequency close to the 15-minute boundary rather than every 5 minutes (fewer requests against your instance-hour budget isn't a factor here since instance-hours are billed on wall-clock uptime, not request count — but fewer pings is still marginally less noise in logs).

Don't point the cron job at `/` — while spun down, Render auto-responds to `/robots.txt` without waking the instance, but every other path (including `/`) does trigger a wake-up and gets routed through your app; using `/health` specifically means the check also validates the model actually loaded, not just that the process is up.

## 9. Post-migration checklist

- [ ] `render.yaml` deployed, service shows **Live** in the Render dashboard
- [ ] `/health` returns `model_loaded: true`
- [ ] `/predict/xgboost`, `/dashboard`, `/docs` all verified against the `*.onrender.com` URL
- [ ] External cron job created and its first few runs show `200` from `/health`
- [ ] Custom domain (if any) re-pointed via DNS to Render
- [ ] AWS App Runner service and (optionally) ECR repo decommissioned
- [ ] AWS secrets removed from GitHub Actions
- [ ] Render workspace's **Billing → Included Usage** checked after a few days to confirm instance-hour consumption matches expectations

## 10. Rollback

Render keeps your two most recent previous deploys available for **Rollback** from the service's **Deploys** tab (free-plan-supported feature) — if a deploy breaks something, roll back with one click while you fix `main`. Because App Runner and ECR won't be deleted until step 7's checklist is complete, you also have the option to point traffic back at the old App Runner URL during the transition if you're running both in parallel temporarily.
