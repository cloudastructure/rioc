# How CI Works

This document describes the continuous integration and continious delivery pipelines for the `rioc` service. The pipeline is defined in [`Jenkinsfile-CI.groovy`](./Jenkinsfile-CI.groovy) and runs on Jenkins.

CI in this repo does **build** only. Deployment to **Kubernetes** is handled by a separate CD pipeline - **CD-general-job**.

---

## TL;DR

On every triggered build, Jenkins:

1. Checks out the requested branch of `rioc` repo.
2. Clones a shared scripts repo (`cloudastructure/cloud-infrastructure`) to get build scripts and Helm charts.
3. Builds the Docker image from `Dockerfile` file.
4. Pushes the image to the GCE container registry, tagged with the short git SHA.
5. Posts STARTED / SUCCESS / FAILED notifications to Slack.

---

## Pipeline parameters

| Parameter | Default | Purpose |
|---|---|---|
| `branch` | `main` | Branch of `rioc` to build. Manual builds can target any branch. |
| `SCRIPTS_REPO_BRANCH` | `kube/base` | Branch of `cloud-infrastructure` to pull shared CI scripts and Helm values from. |
| `HELM_DRY_RUN_DEBUG` | `false` | Only relevant if/when the (currently commented-out) Helm deploy stage is re-enabled. |

---

## What gets built (the image itself)

The repo-root [`Dockerfile`](./Dockerfile) is a hardened multi-stage build:

**Builder stage** (`python:3.12-slim-bookworm`):
- Installs build deps: `build-essential`, `portaudio19-dev`, `libsndfile1`, `libgl1`, `libglib2.0-0`.
- Creates `/opt/venv` and installs CPU-only PyTorch from the PyTorch CPU index (`torch==2.4.1+cpu`, `torchvision==0.19.1+cpu`), then `requirements.txt` on top.
- The `+cpu` wheels are not on PyPI — they live only on `download.pytorch.org/whl/cpu`, so the `--index-url` flag is required.

**Runtime stage** (`python:3.12-slim-bookworm`):
- Runtime libs only: `ffmpeg`, `libgl1`, `libglib2.0-0`, `libsndfile1`, `libportaudio2`, `tini`, `curl`, `ca-certificates`.
- Creates a non-root `rioc` user/group with UID/GID 1000.
- Copies `/opt/venv` from the builder, then app sources (`*.py`, `scripts/`, `mediamtx.yml`).
- Symlinks `/data/ai_guard.db` into the app dir so SQLite state lives on a mounted PV in k8s.
- `tini` is PID 1 to forward SIGTERM cleanly to uvicorn.
- Exposes port 8000; `HEALTHCHECK` polls `GET /events`.
- Default command: `uvicorn main:app --host 0.0.0.0 --port ${PORT} --limit-concurrency 100`.

Vision inference (MiniCPM-o / MiniCPM-V) is offloaded to a remote vLLM server, which is why the image is CPU-only and stays small.

[`.dockerignore`](./.dockerignore) excludes `.git`, `__pycache__/`, `venv/`, `.env*` examples, `data/`, logs, audio files, and the `.continue/` IDE config to keep the build context minimal.

---

## Triggering a build

- **Default (no params)**: builds `main`, image tagged with the `main` HEAD short SHA.
- **Manual / feature branch**: set `branch` to the branch name. The pipeline re-checks out that branch and computes `IMAGE_TAG` from it. Useful for testing branches like `dockerize` before merging.
- **Scripts repo override**: bump `SCRIPTS_REPO_BRANCH` if testing changes to shared CI tooling in `cloud-infrastructure`.

The resulting image lives in the GCE registry and is tagged with the short SHA, so the deploying CD pipeline can pin to a specific commit.

---

## Files involved

| Path | Role |
|---|---|
| `Jenkinsfile-CI.groovy` | The active CI pipeline (this document describes it). |
| `Dockerfile` | Built and pushed by CI. |
| `.dockerignore` | Trims the build context. |
| `scripts/` | Ships inside the image (e.g. `videodb_rtsp.sh`). Do not confuse with `ci-scripts/`, which is pulled from the shared repo at build time. |

External:

| Repo | Used for |
|---|---|
| `cloudastructure/cloud-infrastructure` (`kube/base` by default) | `scripts/groovy/Utils.groovy` (Slack helpers), `scripts/custom-docker-*.sh` (build + push), `helms/` (used only by the disabled deploy stage). |

---

# How CD Works

Deploy is handled by the **`CD-general-job`** Jenkins pipeline. It is a shared job — the same pipeline deploys every Cloudastructure service (account-data, catalog-service, rioc, video-aggregator, …); the `APP_NAME` parameter picks which one.

Pipeline definition: [`JenkinsPipelines/CD-general.groovy`](../cloud-infrastructure/JenkinsPipelines/CD-general.groovy) in the `cloudastructure/cloud-infrastructure` repo.

---

## TL;DR

On each deploy, Jenkins:

1. Checks out `cloud-infrastructure` (branch from `SCRIPTS_REPO_BRANCH`) to get the deploy scripts and Helm charts.
2. Verifies that the image `${APP_NAME}:${image_tag}` exists in the source registry.
3. **Promotes** (re-tags/copies) the image into the target environment's registry.
4. Renders the Helm chart from `helms/deploy-app-helm` with the environment's values file (`helms/values/<env>.yaml`) and deploys it to the target GKE cluster.
5. Posts STARTED / SUCCESS / FAILED Slack notifications (with action label `PROMOTE`).

There is **no rebuild** here — CD reuses the exact image that CI pushed. That's why `image_tag` (the short git SHA produced by CI) is the linkage between the two pipelines.

---

## Pipeline parameters

| Parameter | Choices / Default | Purpose |
|---|---|---|
| `APP_NAME` | choice from a fixed list (includes `rioc`) | Which service to deploy. Picks both the image name and the Helm release. |
| `environment` | `dev-ovh`, `qa-ovh`, `demo-new`, `prod-new` | Target environment. Selects the destination registry, the Helm values file, and the GKE cluster context. |
| `image_tag` | string, no default | The short git SHA from a successful CI build. Required — must match a tag that already exists in the source registry. |
| `SCRIPTS_REPO_BRANCH` | `kube/base` | Branch of `cloud-infrastructure` to use for scripts + Helm charts. |
| `HELM_DRY_RUN_DEBUG` | `false` | If true, `custom-deploy-helm.sh` should `helm --dry-run` instead of applying. Useful for previewing rendered manifests. |

---

## How CI and CD link together

```
   CI (Jenkinsfile-CI.groovy in rioc)            CD (CD-general.groovy in cloud-infrastructure)
   ──────────────────────────────────            ──────────────────────────────────────────────
   build  ──►  push to registry        check-image  ──►  promote to <env> registry  ──►  helm upgrade --install
              tag = <short git SHA>                           ▲
                                                              │
                                                  image_tag parameter (same short SHA)
```

The contract is just the image tag: CI produces `${APP_NAME}:<short-sha>` in the registry; CD takes that same SHA as input, promotes the image to the target environment, and rolls it out via Helm.

---

## Triggering a deploy

1. Wait for a CI build of `rioc` to succeed and note the `image_tag` from the build name or Slack message.
2. Open `CD-general-job` in Jenkins and **Build with Parameters**:
   - `APP_NAME` = `rioc`
   - `environment` = target environment (e.g. `qa-ovh`)
   - `image_tag` = the short SHA from CI
   - leave `SCRIPTS_REPO_BRANCH` at `kube/base` unless testing deploy-script changes
   - set `HELM_DRY_RUN_DEBUG` = true to preview without applying
3. Watch the Slack channel for `PROMOTE STARTED` → `SUCCESS` / `FAILED`.

---

## Result

Once the CD pipeline finishes successfully, the application starts running on one of the Kubernetes clusters mapped to the selected `environment`:

- `dev-ovh` / `qa-ovh` → the OVH-hosted dev / QA clusters
- `demo-new` / `demo-dc-01` → the demo clusters
- `prod-new` → the production cluster

`custom-deploy-helm.sh` issues `helm upgrade --install` against that cluster, so the result is a running `rioc` Deployment (with the freshly promoted image), its Service, and any other resources the shared `deploy-app-helm` chart renders for the chosen environment. Pod readiness is gated by the `HEALTHCHECK` defined in the Dockerfile (`GET /events` on port 8000) — once the readiness probe passes, the service starts taking traffic and the rollout is considered complete.

From this point on, the new build is live and serving the environment until the next CD run promotes a different `image_tag`.