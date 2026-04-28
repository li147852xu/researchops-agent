# Deploy ResearchOps to HuggingFace Spaces

This guide takes you from "empty HF account" to a running Space at
**`https://huggingface.co/spaces/Tiantanghuaxiao/researchops`** in about 30 minutes.

The deployment uses the **Docker SDK**: HF Spaces builds the `Dockerfile` at the
Space repo root and runs the resulting container. ResearchOps ships everything
needed in [`deploy/hf_spaces/`](.). The only twist is that HF Spaces requires
the `Dockerfile` and frontmatter `README.md` to live at the repo root, while in
this monorepo they live in this subfolder — step 2 below shows the one-line
sync that handles it.

---

## Prerequisites

- A free HuggingFace account: https://huggingface.co/join
- A user access token with **write** permission: https://huggingface.co/settings/tokens
- Docker installed locally (only needed for the optional smoke test)
- `git` and `git-lfs` (`brew install git-lfs` / `apt install git-lfs`)

---

## Step 1 — Smoke-test locally (recommended, 3 min)

Before pushing to HF, confirm the image builds and the UI comes up on your
machine. Run this from the **repo root**:

```bash
docker build -t researchops-demo -f deploy/hf_spaces/Dockerfile .
docker run --rm -p 7860:7860 -e LLM_BACKEND=none researchops-demo
```

After the container logs `Running on local URL:  http://0.0.0.0:7860`, open
http://localhost:7860 — you should see three tabs (General Research, Market
Intelligence, Architecture) and be able to run a demo topic in rule-based mode.

If the build fails, fix it locally first; HF Spaces gives the same error
messages but with a slower iteration loop.

---

## Step 2 — Create the Space (2 min)

1. Go to https://huggingface.co/new-space.
2. Fill in:
   - **Owner**: `Tiantanghuaxiao`
   - **Space name**: `researchops`
   - **License**: MIT
   - **Space SDK**: **Docker** → choose **Blank** (we'll provide the Dockerfile).
   - **Hardware**: `CPU basic · 2 vCPU · 16 GB · FREE`
   - **Visibility**: Public (or Private — both work).
3. Click **Create Space**. You now have an empty git repo at
   `https://huggingface.co/spaces/Tiantanghuaxiao/researchops`.

---

## Step 3 — Sync the code & push (10 min)

The HF Space repo expects the Dockerfile and frontmatter README at its root.
This snippet clones the empty Space repo, mirrors the ResearchOps source into
it, hoists the `deploy/hf_spaces/` files to the root, and pushes.

```bash
# From wherever your researchops checkout lives
RESEARCHOPS_DIR="$(pwd)"

# 1) Clone the empty Space
cd /tmp
git clone https://huggingface.co/spaces/Tiantanghuaxiao/researchops hf-space
cd hf-space

# 2) Mirror the ResearchOps source into the Space repo
rsync -av --delete \
  --exclude='.git' --exclude='runs' --exclude='experiments' \
  --exclude='__pycache__' --exclude='.venv' --exclude='.pytest_cache' \
  --exclude='.ruff_cache' --exclude='*.egg-info' \
  "$RESEARCHOPS_DIR"/ ./

# 3) Promote the HF-specific files to the Space repo root
cp deploy/hf_spaces/Dockerfile      ./Dockerfile
cp deploy/hf_spaces/README.md       ./README.md          # overrides project README in the Space
cp deploy/hf_spaces/.env.template   ./.env.template
cp deploy/hf_spaces/DEPLOY.md       ./DEPLOY.md

# 4) Commit & push
git lfs install
git add .
git commit -m "Initial deploy: ResearchOps Gradio UI"
git push
```

When prompted for credentials, the username is your HF handle
(`Tiantanghuaxiao`) and the password is your **access token** from
https://huggingface.co/settings/tokens (not your account password).

> **Note** — the Dockerfile's `CMD` invokes `deploy/hf_spaces/app.py`, which is
> still present in the synced tree at that path. Do not delete the
> `deploy/hf_spaces/` directory in the Space repo.

---

## Step 4 — Configure Secrets (3 min)

If you want the demo to use a real LLM (otherwise skip — `LLM_BACKEND=none` is
the default and the UI works without any Secrets):

1. Open `https://huggingface.co/spaces/Tiantanghuaxiao/researchops/settings`.
2. Scroll to **Variables and secrets** → click **New secret**.
3. Add the variables you need. Minimum for DeepSeek:

   | Name | Value |
   |---|---|
   | `LLM_BACKEND` | `openai_compat` |
   | `LLM_MODEL` | `deepseek-chat` |
   | `LLM_BASE_URL` | `https://api.deepseek.com/v1` |
   | `DEEPSEEK_API_KEY` | `sk-...` (your DeepSeek key) |

   Optional: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`.

4. Click **Save**, then **Restart this Space** so the container picks up the
   new env. See [`.env.template`](.env.template) for every supported variable.

---

## Step 5 — Wait for the build & verify (10–15 min the first time)

1. Click the **Logs** tab on the Space page. Expect:
   - **Build**: ~5 min (pulls the Python base, installs deps).
   - **Container start**: ~30 s (Gradio boots).
2. The build finishes when you see
   `Running on local URL:  http://0.0.0.0:7860`.
3. Open `https://huggingface.co/spaces/Tiantanghuaxiao/researchops`. You should
   see:
   - Header banner with `ResearchOps v1.0.0` and the architecture badges.
   - Three tabs: General Research / Market Intelligence / Architecture.
   - In **General Research**, the topic dropdown lists the 6 demo topics; click
     one (e.g. "transformer architecture evolution"), keep `Mode: fast`, and
     click **Run General Research**.
   - The **Pipeline Stages** indicator should walk through PLAN → COLLECT → READ
     → VERIFY → WRITE → QA → EVAL.
   - The **Report** sub-tab shows the generated Markdown with numbered
     citations.

If nothing renders after 60 s, open the Logs tab — most failures are missing
Secrets or a stale Restart (re-trigger via Settings → "Factory rebuild").

---

## Updating the Space later

Once the Space repo exists, subsequent updates from your local ResearchOps
checkout are a re-run of Step 3:

```bash
cd /tmp/hf-space
rsync -av --delete \
  --exclude='.git' --exclude='runs' --exclude='experiments' \
  --exclude='__pycache__' --exclude='.venv' --exclude='.pytest_cache' \
  --exclude='.ruff_cache' --exclude='*.egg-info' \
  "$RESEARCHOPS_DIR"/ ./
cp deploy/hf_spaces/Dockerfile    ./Dockerfile
cp deploy/hf_spaces/README.md     ./README.md
cp deploy/hf_spaces/.env.template ./.env.template
cp deploy/hf_spaces/DEPLOY.md     ./DEPLOY.md
git add . && git commit -m "Sync from researchops" && git push
```

HF Spaces detects the push, rebuilds, and hot-swaps the running container.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Build failed: pip install ...` | Missing `pyproject.toml` in Space root | Make sure step 3's `rsync` completed; the Space repo must contain the entire ResearchOps source tree, not just the Dockerfile. |
| Build succeeds but UI never loads | Dockerfile `CMD` not running | Confirm `deploy/hf_spaces/app.py` is present in the Space repo (Files tab → search). |
| `503 Service Unavailable` from HF | Container crashed | Check Logs tab. Most often a missing Secret when `LLM_BACKEND` is set but the matching API key isn't. |
| UI shows "Error: ..." after Run | LLM call failed | Either set the right `LLM_BACKEND` + key in Secrets, or unset `LLM_BACKEND` to fall back to rule-based mode. |
| Out-of-memory at runtime | Free tier hit | Ensure the `embeddings` extra is NOT installed (default Dockerfile does not install it). Avoid running multiple "deep" runs in parallel. |
