"""HuggingFace Spaces (Docker SDK) entrypoint for ResearchOps.

Why this file exists
--------------------
``researchops.web.app.launch()`` rewrites ``host=0.0.0.0`` to ``127.0.0.1`` to
keep the local-dev experience friendly behind corporate proxies. Inside a
container that's exactly the wrong default — port mappings (``-p 7860:7860``)
require the server to listen on the wildcard interface. This entrypoint
sidesteps that helper by building the Gradio app via ``create_app()`` and
calling ``demo.launch()`` directly with the explicit container-friendly host.
"""

from __future__ import annotations

import os

from researchops.web.app import create_app


def _normalize_loopback_no_proxy() -> None:
    """Ensure NO_PROXY allows loopback access — kept in sync with researchops.web.app.launch."""
    loopback = "localhost,127.0.0.1,0.0.0.0"
    for env_key in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(env_key, "")
        if existing:
            merged = {v.strip() for v in existing.split(",") if v.strip()} | set(loopback.split(","))
            os.environ[env_key] = ",".join(sorted(merged))
        else:
            os.environ[env_key] = loopback


def main() -> None:
    _normalize_loopback_no_proxy()

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))

    print(f"[researchops.hf_spaces] Booting Gradio UI on http://{host}:{port}")
    print(
        f"[researchops.hf_spaces] LLM_BACKEND={os.environ.get('LLM_BACKEND', 'none')} "
        f"LLM_MODEL={os.environ.get('LLM_MODEL', '(default)')}"
    )

    demo = create_app()
    demo.launch(
        server_name=host,
        server_port=port,
        share=False,
        show_api=False,
        quiet=False,
    )


if __name__ == "__main__":
    main()
