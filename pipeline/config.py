"""The work directory, execution limits, and the model profiles.

The active model is `EMSE_MODEL_PROFILE` (in `.env` or the shell); it is the one
served, scored for perplexity, and whose task outputs are read and written.

    python -m pipeline.config {model-name|serve-args}   # used by `make serve`
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import dotenv

from pipeline.data import DATA, LANGS, ROOT  # noqa: F401

dotenv.load_dotenv(ROOT / ".env")
WORK_DIR = Path(os.environ.get("EMSE_WORK_DIR", "/tmp/emse_work"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

LANGNAME = {"cpp": "C++", "java": "Java"}

# ------------------------------------------------------------------ model ----
VLLM_BASE_URL = "http://127.0.0.1:8123/v1"
MAX_TOKENS = 32768                      # per reply, reasoning included
TRUNCATION_RETRIES = 3


@dataclass(frozen=True)
class ModelProfile:
    model_name: str
    sampling: dict                      # OpenAI-API sampling parameters
    extra_body: dict                    # vLLM-only sampling and chat-template settings
    serve_args: str                     # vLLM flags after `--model <name> --host --port`


# Sampling is each model card's recommended preset for reasoning/coding.
MODEL_PROFILES: dict[str, ModelProfile] = {
    "qwen3.6-35b": ModelProfile(
        model_name="Qwen/Qwen3.6-35B-A3B",
        sampling={"temperature": 0.6, "top_p": 0.95, "presence_penalty": 0.0},
        extra_body={"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0},
        serve_args=("--max-model-len 65536 --gpu-memory-utilization 0.95 "
                    "--reasoning-parser qwen3 --language-model-only"),
    ),
    "gemma-4-26b": ModelProfile(
        model_name="google/gemma-4-26B-A4B-it",
        sampling={"temperature": 1.0, "top_p": 0.95, "presence_penalty": 0.0},
        # The gemma4 reasoning parser needs the special tokens to split reasoning from the answer.
        extra_body={"top_k": 64, "min_p": 0.0, "repetition_penalty": 1.0,
                    "chat_template_kwargs": {"enable_thinking": True}, "skip_special_tokens": False},
        serve_args=("--max-model-len 65536 --gpu-memory-utilization 0.90 "
                    "--reasoning-parser gemma4 --language-model-only"),
    ),
    "nemotron-3-nano-30b-a3b": ModelProfile(
        model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        sampling={"temperature": 1.0, "top_p": 1.0},
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        serve_args=("--max-model-len 65536 --gpu-memory-utilization 0.95 "
                    "--trust-remote-code "
                    "--reasoning-parser-plugin {plugin} --reasoning-parser nano_v3"),
    ),
}

ACTIVE_PROFILE = os.environ.get("EMSE_MODEL_PROFILE", "nemotron-3-nano-30b-a3b")
PROFILE = MODEL_PROFILES[ACTIVE_PROFILE]


def serve_args() -> str:
    """The active `serve_args`, with `{plugin}` resolved to the parser file in the local HF cache."""
    if "{plugin}" not in PROFILE.serve_args:
        return PROFILE.serve_args
    from huggingface_hub import try_to_load_from_cache
    plugin = try_to_load_from_cache(PROFILE.model_name, "nano_v3_reasoning_parser.py")
    return PROFILE.serve_args.format(plugin=plugin)


# --------------------------------------------------------------- execution ---
COMPILE_TIMEOUT = 60.0
RUN_TIMEOUT = 8.0
RUN_MEMORY_MB = 2048
MAX_OUTPUT_BYTES = 1 << 20


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["model-name", "serve-args"])
    print(PROFILE.model_name if ap.parse_args().what == "model-name" else serve_args())
