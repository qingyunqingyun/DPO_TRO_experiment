#!/usr/bin/env python3
"""Build the full eligible-set oracle training dataset from Skywork scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from datasets import Dataset, DatasetDict


def read(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dataset-output", required=True)
    ap.add_argument("--temperature", type=float, default=None)
    args = ap.parse_args()
    pairs_path, scores_path, out = map(Path, (args.pairs, args.scores, args.output))
    pairs = {x["pair_id"]: x for x in read(pairs_path)}
    scores = read(scores_path)
    if set(pairs) != {x["pair_id"] for x in scores}:
        raise RuntimeError("pair IDs in input and score manifests do not match exactly")
    gaps = [abs(float(x["score_1"]) - float(x["score_2"])) for x in scores]
    nonzero = [x for x in gaps if x > 0]
    temperature = args.temperature if args.temperature is not None else statistics.median(nonzero)
    if not math.isfinite(temperature) or temperature <= 0:
        raise RuntimeError(f"invalid temperature: {temperature}")
    records = []
    for score in scores:
        p = pairs[score["pair_id"]]
        gap = float(score["score_1"]) - float(score["score_2"])
        p_star = 1.0 / (1.0 + math.exp(-gap / temperature))
        records.append({
            "pair_id": p["pair_id"], "prompt_id": p["prompt_id"],
            "prompt": p["prompt"], "response_1": p["response_1"],
            "response_2": p["response_2"], "score_1": float(score["score_1"]),
            "score_2": float(score["score_2"]), "p_star": p_star,
            "temperature": temperature,
            "temperature_rule": "median_nonzero_absolute_gap_on_full_train_scores",
        })
    write(out, records)
    method = []
    for r in records:
        method.append({
            "pair_id": r["pair_id"], "prompt_id": r["prompt_id"],
            "prompt": r["prompt"], "chosen": r["response_1"],
            "rejected": r["response_2"], "n_votes": 0, "k_1": 0,
            "p_hat": r["p_star"], "hard_label": 1 if r["p_star"] >= 0.5 else 2,
            "lower": r["p_star"], "upper": r["p_star"], "lambda_i": 1.0,
            "p_star": r["p_star"],
        })
    ds = Dataset.from_list(method)
    # The trainer expects train/dev named splits even though this full oracle
    # run does not use dev for model selection. Keep a deterministic small
    # validation view so the loader and training command are identical to v5.
    dev = ds.select(range(min(500, len(ds))))
    DatasetDict({"train": ds, "dev": dev}).save_to_disk(str(Path(args.dataset_output)))
    meta = {
        "pairs": len(records), "scores": len(scores), "temperature": temperature,
        "temperature_rule": "median_nonzero_absolute_gap_on_full_train_scores",
        "p_star_mean": statistics.mean(r["p_star"] for r in records),
        "p_star_min": min(r["p_star"] for r in records),
        "p_star_max": max(r["p_star"] for r in records),
        "pairs_sha256": stable_hash(pairs_path), "scores_sha256": stable_hash(scores_path),
        "oracle_sha256": stable_hash(out), "dataset_output": str(args.dataset_output),
    }
    (out.parent / "build_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
