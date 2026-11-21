#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path('/mnt/hezhisheng/su/oracle_full_20261120')

def read(path):
    with path.open() as f:
        return [json.loads(x) for x in f if x.strip()]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

rows = read(ROOT / 'eval/full_test_scored.jsonl')
groups = defaultdict(list)
for r in rows:
    groups[str(r['prompt_id'])].append(float(r['skywork_score']))
means = {k: statistics.mean(v) for k, v in groups.items()}
summary = {
    'model': str(ROOT / 'outputs/full_oracle/final_hf'),
    'evaluation_pairs': 1995,
    'evaluation_prompts': len(groups),
    'responses_per_prompt': 10,
    'scored_rows': len(rows),
    'mean_rm_reward_R': statistics.mean(means.values()),
    'prompt_rm_sd': statistics.stdev(means.values()) if len(means) > 1 else 0.0,
    'min_prompt_mean': min(means.values()),
    'max_prompt_mean': max(means.values()),
    'generation_seed': 401,
    'temperature': 0.7,
    'top_p': 0.9,
    'max_new_tokens': 1024,
    'context': 4096,
    'rm': 'Skywork/Skywork-Reward-V2-Llama-3.1-8B',
    'rm_max_length': 4096,
    'train_pairs': 61005,
    'train_max_len': 2048,
    'epochs': 2,
    'learning_rate': 5e-7,
    'beta': 0.01,
    'effective_batch_pairs': 128,
    'oracle_train_manifest': str(ROOT / 'data/oracle_train_full.jsonl'),
    'generated_sha256': sha(ROOT / 'eval/full_test_generations.jsonl'),
    'scored_sha256': sha(ROOT / 'eval/full_test_scored.jsonl'),
}
(ROOT / 'report/full_oracle_metrics.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
with (ROOT / 'report/full_oracle_prompt_means.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['prompt_id', 'mean_rm_reward', 'responses'])
    for k in sorted(means):
        w.writerow([k, means[k], len(groups[k])])
md = '# Full eligible-set oracle result (2026-11-20)\n\n'
md += 'This run trains one `oracle_soft_dpo` Llama-3.2-1B-Instruct policy on all 61,005 length-eligible, prompt-deduplicated training pairs. The 1,995 eligible `test_filtered` prompts are held out for evaluation.\n\n'
md += '## Training and scoring\n\n| item | value |\n|---|---|\n'
for k, v in [
    ('train pairs', 61005), ('test prompts', len(groups)), ('responses per prompt', 10),
    ('epochs', 2), ('max policy length', 2048), ('effective batch', 128),
    ('learning rate', 5e-7), ('DPO beta', 0.01), ('generation temperature', 0.7),
    ('top-p', 0.9), ('max new tokens', 1024), ('RM', 'Skywork-Reward-V2-Llama-3.1-8B')
]:
    md += f'| {k} | {v} |\n'
md += '\n## Result\n\n'
md += f"- Mean within-prompt RM reward `R`: **{summary['mean_rm_reward_R']:.6f}**\n"
md += f"- Across-prompt standard deviation: `{summary['prompt_rm_sd']:.6f}`\n"
md += f"- Scored rows: `{len(rows)}` (`{len(groups)} × 10`)\n\n"
md += '`R` is the mean of the ten-response mean reward for each held-out prompt. This is an oracle-policy reference result; no competing full-data policy was trained in this run.\n\n'
md += '## Artifacts\n\n- `data/oracle_train_full.jsonl`: full p* training manifest\n- `datasets/full_oracle`: HF training dataset\n- `outputs/full_oracle/final_hf`: final policy checkpoint\n- `eval/full_test_generations.jsonl`: all generated responses\n- `eval/full_test_scored.jsonl`: RM scores\n- `report/full_oracle_metrics.json` and `report/full_oracle_prompt_means.csv`: summaries\n'
(ROOT / 'report/FULL_ORACLE_REPORT.md').write_text(md)
print(json.dumps(summary, indent=2, ensure_ascii=False))
