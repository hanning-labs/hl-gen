"""Batch synthesis runner.

Usage
-----
    python examples/run_batch.py configs/default.yaml

Reads the YAML config, randomly samples ``n`` SynthesisRequests, and runs
them concurrently. Accepted samples are written to the path set in ``output``.

LocalClient config (edit the script or subclass BatchConfig to override):
    model           HF model id           default: "Qwen/Qwen2.5-7B-Instruct"
    device          device_map            default: None → "auto"
    dtype           torch_dtype           default: "auto"
    max_new_tokens  generation length     default: 1024
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).parent.parent))

from batch import BatchConfig, default_max_concurrent, run_batch
from llm import LocalClient
from orchestrator import build_default_pipeline
from storage.file_store import FileSampleStore


async def main(config_path: str) -> None:
    with open(config_path) as f:
        config = BatchConfig.model_validate(yaml.safe_load(f))

    n_jobs = config.max_concurrent or default_max_concurrent()

    print(f"Batch synthesis")
    print(f"  config      : {config_path}")
    print(f"  n           : {config.n}")
    print(f"  concurrent  : {n_jobs}")
    print(f"  output      : {config.output}")
    print()

    llm = LocalClient()
    store = FileSampleStore(config.output)
    pipeline = build_default_pipeline(llm, store)

    results = await run_batch(config, pipeline)

    accepted = sum(1 for r in results if r is not None)
    print()
    print(f"Done. {accepted}/{config.n} samples accepted → {config.output}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python examples/run_batch.py <config.yaml>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
