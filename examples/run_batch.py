"""Batch synthesis runner.

Usage
-----
    python examples/run_batch.py configs/default.yaml

Reads the YAML config, randomly samples ``n`` SynthesisRequests, and runs
them concurrently. Accepted samples are written to the path set in ``output``.
If ``output`` already contains accepted samples the run resumes from where it
left off — only the remaining ``n - already_accepted`` attempts are made.

All knobs (model, device, dtype, seed, concurrency, …) live in the YAML config.
See configs/default.yaml for the full reference.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).parent.parent))

from batch import BatchConfig, count_existing, default_max_concurrent, run_batch
from llm import LocalClient
from orchestrator import build_default_pipeline
from storage.file_store import FileSampleStore


async def main(config_path: str) -> None:
    with open(config_path) as f:
        config = BatchConfig.model_validate(yaml.safe_load(f))

    already_done = count_existing(config.output)
    remaining = config.n - already_done
    if remaining <= 0:
        print(f"Already complete: {already_done}/{config.n} samples in {config.output}")
        return

    n_jobs = config.max_concurrent or default_max_concurrent()

    print(f"Batch synthesis")
    print(f"  config      : {config_path}")
    print(f"  target      : {config.n}  (resuming — {already_done} already accepted)")
    print(f"  remaining   : {remaining}")
    print(f"  concurrent  : {n_jobs}")
    print(f"  output      : {config.output}")
    print()

    llm = LocalClient(**config.client.model_dump())
    store = FileSampleStore(config.output)
    pipeline = build_default_pipeline(llm, store)

    effective = config.model_copy(update={"n": remaining})
    results = await run_batch(effective, pipeline, seed=config.seed)

    newly_accepted = sum(1 for r in results if r is not None)
    total_accepted = already_done + newly_accepted
    print()
    print(f"Done. {total_accepted}/{config.n} samples accepted → {config.output}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python examples/run_batch.py <config.yaml>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
