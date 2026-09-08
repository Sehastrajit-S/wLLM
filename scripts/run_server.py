"""Dev convenience wrapper around wllm.cli -- once wLLM is pip-installed
(`pip install -e .` for local dev, or `pip install wllm` for real use), the
`wllm-server` console command (see pyproject.toml's [project.scripts]) is
the actual entry point; this script exists so `python scripts/run_server.py`
also works from a source checkout without installing anything first.

Run: python scripts/run_server.py --model Qwen/Qwen2.5-0.5B-Instruct --port 8000
"""
import sys

sys.path.insert(0, "src")

from wllm.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
