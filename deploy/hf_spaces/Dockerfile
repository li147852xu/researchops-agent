# syntax=docker/dockerfile:1.6
#
# ResearchOps — HuggingFace Spaces (Docker SDK) image.
#
# Build (locally, from repo root):
#   docker build -t researchops-demo -f deploy/hf_spaces/Dockerfile .
#
# Run (locally):
#   docker run --rm -p 7860:7860 -e LLM_BACKEND=none researchops-demo
#
# When deployed to HF Spaces, this Dockerfile is placed at the Space repo root
# (see DEPLOY.md). The Space repo also contains the full ResearchOps source, so
# `pip install -e ".[web]"` resolves to the package at /home/user/app.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    LLM_BACKEND=none \
    HOST=0.0.0.0 \
    PORT=7860 \
    HF_HOME=/home/user/.cache/huggingface \
    XDG_CACHE_HOME=/home/user/.cache

RUN useradd --create-home --uid 1000 user

USER user
ENV PATH="/home/user/.local/bin:${PATH}"

WORKDIR /home/user/app

COPY --chown=user:user . /home/user/app/

RUN pip install --user --no-cache-dir --upgrade pip \
 && pip install --user --no-cache-dir -e ".[web]"

EXPOSE 7860

CMD ["python", "deploy/hf_spaces/app.py"]
