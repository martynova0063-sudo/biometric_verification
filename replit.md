# ИС УКЗ · HEALTH_ID

Research console for HEALTH_ID calculations, biometric verification, specialist review, and model governance. Research results are not medical diagnoses.

## Run & Operate

- Start the existing `artifacts/api-server: API Server` and `artifacts/ukz-health-id: web` workflows in Replit. These provide the service ports and frontend base path.
- API workflow: `pnpm --filter @workspace/api-server run dev` — applies Alembic migrations, then starts FastAPI on the managed port (8080).
- Web workflow: `pnpm --filter @workspace/ukz-health-id run dev` — starts Vite at `/`.
- `artifacts/mockup-sandbox: Component Preview Server` is an optional design preview, not required by the main app.
- Node dependencies: `pnpm install --frozen-lockfile`.
- Python dependencies are declared in `pyproject.toml` and locked in `uv.lock`; API commands use `uv run`.
- `pnpm run typecheck` — full typecheck across all packages
- Frontend build from the shell: `PORT=20404 BASE_PATH=/ pnpm --filter @workspace/ukz-health-id run build`.
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `DATABASE_URL` is already supplied securely by the environment. FastAPI supports PostgreSQL using psycopg; without this variable, its existing fallback is local SQLite.
- Health check: `/api/healthz`. API docs: `/api/docs`. OpenAPI: `/api/openapi.json`.
- Startup seeds sample records through the existing `seed_demo` behavior. The app requires a Replit OIDC sign-in and stores server-side sessions in PostgreSQL; role enforcement is not configured. Use demo data only until role-based access controls are added; do not publish for real medical data.
- API clients may also call `POST /api/auth/login` with `{ "password": "…" }` and then send `Authorization: Bearer <access_token>`. Configure `AUTH_LOGIN_PASSWORD` and a separate `AUTH_JWT_SECRET` (at least 32 bytes) in Replit Secrets. Tokens expire after 15 minutes; the shared password grants the same API access as an OIDC session and is for trusted clients only.
- On the embeddings page, an operator can request an on-demand LLM interpretation of an available comparison. Configure `OPENAI_API_KEY` in Replit Secrets; API use is billed by OpenAI. Only aggregate counts, mean/variance summaries, centroid similarity, outlier summary, and histogram overlap are sent—never video, raw vectors, employee identifiers, or session IDs. The cached interpretation is technical guidance only and does not change the verification decision; refreshing makes another paid call.

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- Frontend: React, Vite, Tailwind CSS, TanStack Query
- API: Python 3.13, FastAPI, Uvicorn
- Active application DB: PostgreSQL + SQLAlchemy, Alembic migrations
- Biometric video processing: OpenCV
- API validation: Pydantic
- API codegen: Orval (from OpenAPI spec)

## Where things live

- `artifacts/ukz-health-id/src/` — application UI and styles.
- `artifacts/api-server/app/` — FastAPI endpoints, models, calculation and verification logic.
- `artifacts/api-server/alembic/` — application database migrations.
- `lib/api-spec/` and `lib/api-client-react/` — API contract and generated frontend client.
- `artifacts/mockup-sandbox/` — optional design component previews.

## Model Configuration

- System defaults and parameter metadata come from `verification_config.py`; the persisted registry stores versioned copies.
- Only the model marked `current` supplies settings to new biometric verification sessions. Draft models do not affect verification until activated.
- Verification records keep their model version and configuration fingerprint for audit. Archiving is reversible, but the active model must be replaced before it can be archived.

## Product

- Dashboard, HEALTH_ID calculation and result history, photo/video verification, specialist review queue, model card and version history.
- Employee directory imported idempotently from `artifacts/api-server/data/employee.json` at API startup; the `employees` table stores employee attributes and each reference photo is packaged under `artifacts/api-server/data/employees/<guid>/`.
- The employee selector groups test employees and company employees, sorts each group by full name, and uses the selected personnel number as `employee_ref`. Reference photos are loaded automatically; the existing `POST /api/verification/sessions` multipart fields remain unchanged.
- The supplied employee file has 11 records, while the original photo archive contains photos for 8. In Verification, an authenticated operator can capture or replace an employee's JPEG reference photo; the current private object path is stored in `employees.photo_path`, and imports preserve it. On replacement, the prior photo is copied into the private `dataset/employees/<guid>/` collection and indexed in `employee_photo_archives` with its source path, checksum, size, timestamp, and operator. Neither the original object nor its archive copy is deleted.

## Biometric Quality Metrics

- New verification sessions persist the model's `is_match`, `is_live`, raw cosine score, and UTC `created_at` independently of operator labels.
- `ground_truth_match` and `ground_truth_liveness` are written only when a specialist submits a manual decision; the model's original outputs are not overwritten.
- `POST /api/verification/metrics/calculate?window_days=7` calculates and stores FAR, FRR, CER, APCER, BPCER, and ACER. The governance page provides the same action and a saved history.
- Uploaded reference photos and videos are not retained by the API. The review UI can show them only from the same open browser tab that created the session; they are lost on reload.
- The field `cosine_distance` currently stores the analyzer's raw cosine similarity score (also shown as `match_score`), not `1 - similarity`.

## ArcFace Employee Search

- The Verification page includes single-photo ArcFace candidate search, a protected `POST /api/recognize` endpoint, and `POST /api/refresh-index` to reload the FAISS index without restarting the API.
- Search returns up to five distinct employee UUIDs with cosine similarity. The default `0.7` threshold is a configurable research threshold, not an access decision; use the separate video/liveness verification and human review flow.
- Uploaded query images are processed in memory and are not retained. Requests require an authenticated session.
- The Verification page can enroll a saved employee reference photo with `POST /api/employees/{employee_guid}/arcface-index`. It computes one embedding, stores it in the enrollment sidecar arrays, reloads it into FAISS, and treats an identical embedding as already enrolled.
- A verification can explicitly add selected ResNet50 + ArcFace video embeddings to the FAISS search index, but only after a successful session. Video embeddings are stored separately from manual photo enrollment, deduplicated, and are not attached to verification-session records.
- Enrollment sidecars are loaded at startup and by `POST /api/refresh-index`; `GET /api/arcface/status` reports total and video-vector counts. They are not overwritten by a full `train_and_embed.py` rebuild.
- The `resnet50_arcface` verification setting uses the project's trained `app/storage/best_arcface_resnet50.pth` checkpoint; `arcface` continues to use DeepFace ArcFace.
- To enroll new employees, add their face photos under `artifacts/api-server/data/arcface_dataset/<uuid>/`, then run `cd artifacts/api-server && uv run python -m app.train_and_embed`. The script rebuilds all vectors in `app/storage`; call `POST /api/refresh-index` afterward.
- `class_mapping.json` describes the checkpoint's training classes. `employee_uuids.npy` is the row-by-row identity mapping for the vector index, including newly enrolled employees. The script does not fine-tune the model weights.

## Gotchas

- Filtered pnpm scripts run inside the selected package directory. API migration paths are relative to `artifacts/api-server`.
- Frontend `PORT` and `BASE_PATH` are required by Vite and injected by its managed workflow.
- Requests to `/api` are handled by Replit's shared proxy; no separate Vite API proxy is needed.
- The active Python schema is managed by Alembic, not the unused template Drizzle package.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
