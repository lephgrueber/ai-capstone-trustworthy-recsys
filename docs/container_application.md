# Full application containers

The full application consists of a Linux API image and a production frontend image. `docker-compose.yml` remains the standalone Weaviate definition used by CI and existing database commands. `compose.application.yml` deliberately contains no Weaviate service: it joins the already-created Weaviate network and cannot create, replace, or delete that database volume.

The tested target is native `linux/arm64`. Other platforms remain unverified. The API uses Python 3.13.7, CPU-only PyTorch 2.10.0, the direct dependency pins in `constraints/application-linux-py313.txt`, and Debian `libgomp1`. The frontend build uses Node 24.11.1 with the npm shipped by that image; nginx-unprivileged 1.29.4 Alpine serves the resulting static files. No host `.venv`, `node_modules`, model, data, credential, or feedback database enters either image.

## A. Start with existing compatible assets

This path is for an existing populated collection and matching saved artifacts. It does not train or publish anything.

First inspect the existing database identity. Replace the example container name only if inspection shows a different active service:

```bash
docker inspect ai-capstone-trustworthy-recsys-weaviate-1 \
  --format 'project={{index .Config.Labels "com.docker.compose.project"}} image={{.Config.Image}} mounts={{json .Mounts}} networks={{json .NetworkSettings.Networks}}'
docker volume inspect ai-capstone-trustworthy-recsys_weaviate_data
docker network inspect ai-capstone-trustworthy-recsys_default
```

Copy the example and set absolute host paths to the complete compatible directories:

```bash
cp .env.application.example .env.application
```

The prepared run must include `ratio_80_10_10/train.parquet`; LambdaRank reconstructs feature tables from the training interactions. Mounting only `movies.parquet` and weights is insufficient. Keep the reranker directory basename `lambdarank_v1`, which contributes to its reported version.

Build with a real Linux dependency installation, then start on an isolated origin:

```bash
docker compose --env-file .env.application -f compose.application.yml build --pull
APPLICATION_PORT=15173 docker compose --env-file .env.application -f compose.application.yml up -d
docker compose --env-file .env.application -f compose.application.yml ps
```

Startup waits for Weaviate before loading the API. The API then performs its existing manifest, model/data, saved-vector, remote collection, and LambdaRank compatibility audits. The health check allows three minutes for those audits. It proves bounded startup readiness, not recovery from a later database outage.

Check the production frontend proxy, learned identity, and a representative learned request:

```bash
curl -fsS http://127.0.0.1:15173/ready
curl -fsS http://127.0.0.1:15173/api/v1/application/status
curl -fsS -X POST http://127.0.0.1:15173/api/v1/recommendations \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"container-smoke","preference_revision":0,"preferences":{"history_items":[],"excluded_movie_ids":[],"include_genres":[],"exclude_genres":[],"ratings":[]},"temporary_history_items":[],"result_count":3,"known_user_id":null}'
```

Read sanitized service logs. The nginx format omits query strings and request bodies:

```bash
docker compose --env-file .env.application -f compose.application.yml logs --tail=200 api frontend
```

After isolated verification, identify the exact process using the normal frontend port, stop only that old application process, and start the containers on the normal origin:

```bash
lsof -nP -iTCP:5173 -sTCP:LISTEN
APPLICATION_PORT=15173 docker compose --env-file .env.application -f compose.application.yml stop frontend
APPLICATION_PORT=5173 docker compose --env-file .env.application -f compose.application.yml up -d frontend
curl -fsS http://127.0.0.1:5173/ready
```

Opening `http://127.0.0.1:5173/` preserves the existing origin and therefore the existing browser-local profile. Storage at port 15173 is a different origin and is not a profile-migration check.

Stop or restart the application without deleting data:

```bash
docker compose --env-file .env.application -f compose.application.yml stop
docker compose --env-file .env.application -f compose.application.yml start
```

Do not use `down -v`, `docker volume rm`, or pruning commands. Plain `down` does not remove the external Weaviate network/volume, but `stop`/`start` is preferred because it also retains the application feedback volume. Feedback is off by default; if explicitly enabled, its SQLite database uses that separate writable volume so WAL/SHM files can be created.

TMDb credentials belong only in ignored `.env.application` runtime configuration. They are passed to the API container and never appear in frontend build variables or image layers.

## B. Genuinely new installation

A new machine still needs a complete prepared run, extracted retrieval inputs, the saved retrieval model, the saved `lambdarank_v1` directory, and a populated Weaviate collection. Container startup does not create any of these.

After following [the trained-model setup guide](team_model_setup.md), publish saved embeddings once with a new output directory and a new collection name:

```bash
docker compose up -d
.venv/bin/python -m trustworthy_recsys.retrieval.publish \
  --inputs-dir data/downloads/two_tower_80_v1/inputs \
  --model-dir data/downloads/two_tower_80_v1/model \
  --output-dir artifacts/retrieval/my_new_weaviate_publication \
  --collection MovieLensMyNewPublication
```

That command imports saved embeddings; it does not train. It refuses an existing output directory or collection name. Never point it at the currently verified collection as a startup step.

## Container verification

Run backend tests in the API test target and frontend checks in the declared Node image:

```bash
docker build --platform linux/arm64 --target test -t trustworthy-recsys-application-api:test -f Dockerfile.api .
docker run --rm --platform linux/arm64 --entrypoint python trustworthy-recsys-application-api:test -m pytest -q
docker build --platform linux/arm64 --target build -t trustworthy-recsys-application-frontend:build -f frontend/Dockerfile .
docker run --rm --platform linux/arm64 trustworthy-recsys-application-frontend:build npm test -- --run
docker run --rm --platform linux/arm64 trustworthy-recsys-application-frontend:build npm run typecheck
docker run --rm --platform linux/arm64 trustworthy-recsys-application-frontend:build npm run build
```

The API `--entrypoint` override is test-only. Normal containers retain the
bounded Weaviate wait and Uvicorn entrypoint declared by `Dockerfile.api`.

With the application already running, the container Playwright configuration never starts or reuses host Vite:

```bash
cd frontend
CONTAINER_APP_URL=http://127.0.0.1:5173 \
  node node_modules/@playwright/test/cli.js test \
  --config=playwright.container.config.ts --project=container-artwork
```

Use the no-artwork project only against an API explicitly started without `TMDB_READ_ACCESS_TOKEN`:

```bash
CONTAINER_APP_URL=http://127.0.0.1:5173 \
  node node_modules/@playwright/test/cli.js test \
  --config=playwright.container.config.ts --project=container-no-artwork
```
