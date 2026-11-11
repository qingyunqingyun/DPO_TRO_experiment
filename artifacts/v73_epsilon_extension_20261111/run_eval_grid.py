#!/usr/bin/env python3
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path('/mnt/hezhisheng/su/experiment_v73_20261111')
OLD = Path('/mnt/hezhisheng/su/1030-lastest/experiment_handoff_v5')
SRC = Path('/mnt/hezhisheng/distributionally_robust_dpo-2')
PY = SRC / '.venv/bin/python'
GEN = OLD / 'generate_eval.py'
SCORE = Path('/mnt/hezhisheng/su/experiment_handoff_v5/score_skywork.py')
PAIRS = ROOT / 'data/pairs/dev.jsonl'
RM = Path('/mnt/hezhisheng/su/experiment_handoff_v5/models/Skywork-Reward-V2-Llama-3.1-8B')
EVAL = ROOT / 'eval'
LOGS = ROOT / 'logs'
EVAL.mkdir(exist_ok=True); LOGS.mkdir(exist_ok=True)

def eval_one(c, gpu):
    name = c['name']; model = ROOT / 'outputs' / name / 'final_hf'
    if not (model / 'config.json').exists():
        model = ROOT / 'outputs' / name / 'checkpoints_dpo' / 'final_hf'
    g = EVAL / name / 'generations.jsonl'; s = EVAL / name / 'scored.jsonl'
    log = LOGS / f'eval_{name}.log'
    EVAL.joinpath(name).mkdir(parents=True, exist_ok=True)
    cmd1 = [str(PY), str(GEN), '--model', str(model), '--pairs', str(PAIRS), '--output', str(g),
                '--k', '10', '--seed', '2101', '--max-new-tokens', '1024', '--max-model-len', '4096', '--gpu-memory-utilization', '0.4']
    cmd2 = [str(PY), str(SCORE), '--model', str(RM), '--input', str(g), '--output', str(s),
                '--mode', 'responses', '--batch-size', '2', '--max-length', '4096', '--device', 'cuda:0']
    env = os.environ.copy(); env.update({'CUDA_VISIBLE_DEVICES': str(gpu), 'VLLM_CONFIGURE_LOGGING': '0', 'PYTHONUNBUFFERED': '1'})
    with log.open('a') as f:
        f.write('GEN: ' + ' '.join(cmd1) + '\nSCORE: ' + ' '.join(cmd2) + '\n')
        f.flush()
        if not g.exists() or sum(1 for _ in g.open()) != 5000:
            p = subprocess.run(cmd1, cwd=str(OLD), env=env, stdout=f, stderr=subprocess.STDOUT)
            if p.returncode:
                return {'name':name, 'generation':str(g), 'scored':str(s), 'complete':False, 'failed_stage':'generation', 'returncode':p.returncode}
        if not s.exists() or sum(1 for _ in s.open()) != 5000:
            p = subprocess.run(cmd2, cwd=str(OLD), env=env, stdout=f, stderr=subprocess.STDOUT)
            if p.returncode:
                return {'name':name, 'generation':str(g), 'scored':str(s), 'complete':False, 'failed_stage':'scoring', 'returncode':p.returncode}
    return {'name': name, 'generation': str(g), 'scored': str(s), 'complete': s.exists() and sum(1 for _ in s.open()) == 5000}

def main():
    m = json.loads((ROOT / 'v7_grid_manifest.json').read_text())
    configs = m['configs']
    # Two single-GPU evaluators; swap to free cards if the cluster allocation changes.
    results=[]
    def worker(lane):
        return [eval_one(c,str([1,4][lane])) for c in configs[lane::2]]
    with ThreadPoolExecutor(max_workers=2) as ex:
        fs=[ex.submit(worker,lane) for lane in range(2)]
        for f in as_completed(fs):
            rs=f.result(); results.extend(rs); print(json.dumps(rs), flush=True)
            (ROOT/'eval_status.json').write_text(json.dumps(results, indent=2))
    results.sort(key=lambda x:x['name']); (ROOT/'eval_status.json').write_text(json.dumps(results, indent=2))
    scored=[r['scored'] for r in results if r['complete']]
    oracle=EVAL/'oracle_soft_dpo/scored.jsonl'
    specs=[f"{r['name']}={r['scored']}" for r in results if r['complete'] and r['name']!='oracle_soft_dpo']
    if oracle.exists() and len(specs)==33:
        cmd=[str(PY), str(OLD/'metrics_shortfall.py'), '--oracle', str(oracle), '--input', *specs, '--output', str(EVAL/'v7_dev_metrics.json'), '--bootstrap', '2000', '--seed', '601']
        with (LOGS/'metrics_v7.log').open('w') as f: subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    sys.exit(0 if len(scored)==34 else 1)
if __name__=='__main__': main()
