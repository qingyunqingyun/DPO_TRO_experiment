#!/usr/bin/env python3
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path('/mnt/hezhisheng/su/experiment_v7_20261110')
SRC = Path('/mnt/hezhisheng/distributionally_robust_dpo-2')
PY = SRC / '.venv/bin/python'
MODEL = SRC / 'models/Llama1b'
LOGS = ROOT / 'logs'
OUTS = ROOT / 'outputs'
LOGS.mkdir(exist_ok=True)
OUTS.mkdir(exist_ok=True)

def finished(out: Path):
    hf = out / 'checkpoints_dpo' / 'final_hf'
    return hf.exists() and any(hf.glob('*.safetensors')) and (hf / 'config.json').exists()

def train_one(c, gpus, port):
    name, method = c['name'], c['method']
    out = OUTS / name
    log = LOGS / f'train_{name}.log'
    status_path = OUTS / name / 'status.json'
    out.mkdir(parents=True, exist_ok=True)
    if finished(out):
        return {'name': name, 'status': 'skipped_existing'}
    cmd = [
        str(PY), '-m', 'deepspeed.launcher.runner', '--include', f'localhost:{gpus}',
        '--master_port', str(port), str(SRC / 'src/train_alignment.py'),
        '--save_path', str(out), '--ckpt_path', str(out / 'checkpoints_dpo'),
        '--save_hf_ckpt', '--disable_ds_ckpt', '--save_steps=-1', '--eval_steps=-1',
        '--micro_train_batch_size=1', '--train_batch_size=128', '--gradient_checkpointing',
        '--zero_stage=2', '--bf16', '--no-flash_attn', '--torch_adam',
        '--learning_rate=5e-7', '--lr_warmup_ratio=0.1', '--max_norm=1.0',
        '--adam_betas', '0.9', '0.95', '--l2=0', '--zpg=2', '--train_task=tro',
        f'--tro_method={method}', '--max_epochs=2', '--beta=0.01',
        '--pretrain', str(MODEL), '--ref_pretrain', str(MODEL), '--dataset', c['dataset'],
        '--prompt_key=prompt', '--chosen_key=chosen', '--rejected_key=rejected',
        '--apply_chat_template', '--max_len=2048', '--dataset_num_processors=1',
        '--train_split=train', '--eval_split=dev', '--seed=1101',
    ]
    rec = {'name': name, 'method': method, 'gpus': gpus, 'port': port,
           'start_time': time.time(), 'command': cmd, 'log': str(log), 'status': 'running'}
    status_path.write_text(json.dumps(rec, indent=2))
    env = os.environ.copy()
    env.update({'DS_SKIP_CUDA_CHECK': '1', 'VLLM_CONFIGURE_LOGGING': '0', 'PYTHONUNBUFFERED': '1'})
    with log.open('w') as f:
        f.write('COMMAND: ' + ' '.join(cmd) + '\n\n')
        f.flush()
        p = subprocess.run(cmd, cwd=str(SRC), env=env, stdout=f, stderr=subprocess.STDOUT)
    rec.update({'end_time': time.time(), 'returncode': p.returncode,
                'status': 'completed' if p.returncode == 0 and finished(out) else 'failed'})
    status_path.write_text(json.dumps(rec, indent=2))
    return rec

def main():
    manifest = json.loads((ROOT / 'v7_grid_manifest.json').read_text())
    configs = manifest['configs']
    # Two independent 2-GPU workers; the other GPUs are occupied by unrelated jobs.
    assignments = [('1,4', 29981), ('5,7', 29982)]
    results = []
    def worker(lane):
        gpu, port = assignments[lane]
        return [train_one(c, gpu, port) for c in configs[lane::len(assignments)]]
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(worker, i) for i in range(len(assignments))]
        for fut in as_completed(futs):
            rs = fut.result(); results.extend(rs)
            print(json.dumps(rs, sort_keys=True), flush=True)
            (ROOT / 'train_status.json').write_text(json.dumps(results, indent=2))
    results.sort(key=lambda x: x['name'])
    (ROOT / 'train_status.json').write_text(json.dumps(results, indent=2))
    failed = [r for r in results if r.get('status') == 'failed']
    print(f'finished={len(results)} failed={len(failed)}', flush=True)
    sys.exit(1 if failed else 0)

if __name__ == '__main__':
    main()
