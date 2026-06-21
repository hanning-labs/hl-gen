"""Batch runner for the English topics synthesis pipeline.

Usage
-----
    python examples/run_topics_batch.py configs/topics_default.yaml

Reads the YAML config, randomly samples ``n`` TopicsRequests grounded in
Currents API articles, and runs them concurrently.
Accepted samples are written to the path set in ``output``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).parent.parent))

from batch import BatchRun, count_existing, default_max_concurrent, run_batch
from config_topics import TopicsBatchConfig, sample_topics_request
from llm import LocalClient
from orchestrator import build_topics_pipeline
from storage.file_store import FileSampleStore
from tools.currents import CurrentsTool


def _write_profile(config_path: str, config: TopicsBatchConfig, n_concurrent: int, batch_run: BatchRun, total_accepted: int) -> Path:
    t = batch_run.request_timings_sec
    sorted_t = sorted(t)
    n = len(t)
    p95 = sorted_t[max(0, int(0.95 * n) - 1)] if n else 0.0
    n_attempted = len(batch_run.results)
    wall = batch_run.wall_sec

    profile = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "pipeline": "topics",
        "model": config.client.model,
        "seed": config.seed,
        "n_attempted": n_attempted,
        "n_accepted": total_accepted,
        "n_rejected": n_attempted - total_accepted,
        "acceptance_rate": round(total_accepted / n_attempted, 4) if n_attempted else 0.0,
        "n_concurrent": n_concurrent,
        "wall_sec": round(wall, 3),
        "throughput_requests_per_min": round(n_attempted / wall * 60, 2) if wall else 0.0,
        "throughput_accepted_per_min": round(total_accepted / wall * 60, 2) if wall else 0.0,
        "request_sec": {
            "mean": round(statistics.mean(t), 3) if t else 0.0,
            "min": round(min(t), 3) if t else 0.0,
            "max": round(max(t), 3) if t else 0.0,
            "p50": round(statistics.median(t), 3) if t else 0.0,
            "p95": round(p95, 3),
        },
    }

    docs = Path(__file__).parent.parent / "docs"
    docs.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = docs / f"profile_topics_{ts}.json"
    out.write_text(json.dumps(profile, indent=2))
    return out


async def main(config_path: str) -> None:
    with open(config_path) as f:
        config = TopicsBatchConfig.model_validate(yaml.safe_load(f))

    already_done = count_existing(config.output)
    remaining = config.n - already_done
    if remaining <= 0:
        print(f"Already complete: {already_done}/{config.n} samples in {config.output}")
        return

    n_jobs = config.max_concurrent or default_max_concurrent()

    print("Batch topics synthesis")
    print(f"  config      : {config_path}")
    print(f"  target      : {config.n}  (resuming — {already_done} already accepted)")
    print(f"  remaining   : {remaining}")
    print(f"  concurrent  : {n_jobs}")
    print(f"  output      : {config.output}")
    print()

    llm = LocalClient(**config.client.model_dump())
    store = FileSampleStore(config.output)
    pipeline = build_topics_pipeline(llm, store, tools=[
        CurrentsTool(categories=config.categories, news_types=config.news_types, max_articles=1)
    ])

    effective = config.model_copy(update={"n": remaining})
    batch_run = await run_batch(effective, pipeline, sample_fn=sample_topics_request, seed=config.seed)

    newly_accepted = sum(1 for r in batch_run.results if r is not None)
    total_accepted = already_done + newly_accepted
    print()
    print(f"Done. {total_accepted}/{config.n} samples accepted → {config.output}")

    profile_path = _write_profile(config_path, config, n_jobs, batch_run, total_accepted)
    print(f"Profile  → {profile_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python examples/run_topics_batch.py <config.yaml>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
