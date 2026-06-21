"""Entry-point shim. The app lives in src/tm_llm/app.py.

Run with:
    uv run uvicorn tm_llm.app:app --reload --host 0.0.0.0 --port 8000
Or directly:
    uv run python main.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tm_llm.app import app  # noqa: F401 — re-exported for uvicorn

if __name__ == "__main__":
    import uvicorn
    from tm_llm.config import PORT
    uvicorn.run(app, host="0.0.0.0", port=PORT)
