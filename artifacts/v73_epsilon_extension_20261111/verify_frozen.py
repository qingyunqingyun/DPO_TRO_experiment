#!/usr/bin/env python3
"""Check content, not timestamps, before reuse of v7.2 artifacts."""
import hashlib, json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'experiment_v7_20261110'
REPO=Path('/mnt/hezhisheng/DPO_TRO_experiment')
SRC=Path('/mnt/hezhisheng/distributionally_robust_dpo-2')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    records=[]
    for f in ['tro_math.py','tro_dataset.py','tro_trainer.py','train_alignment.py','actor.py','drdpo_flash_attn_compat.py']:
        p=SRC/'src'/f
        records.append({'role':'training_source','path':str(p),'sha256':sha(ROOT/'audit'/f),'bytes':p.stat().st_size})
    for source,archive in [(Path('/mnt/hezhisheng/su/1030-lastest/experiment_handoff_v5/generate_eval.py'),'generate_eval.py'),(Path('/mnt/hezhisheng/su/experiment_handoff_v5/score_skywork.py'),'score_skywork.py')]:
        records.append({'role':'evaluation_source','path':str(source),'sha256':sha(ROOT/'audit'/archive),'bytes':source.stat().st_size})
    records+=json.loads((OLD/'audit/initial_model_hashes.json').read_text())
    records += [{'role':'previous_model',**r} for r in json.loads((OLD/'audit/model_exports.json').read_text()) if '_dro_' in r['config'] or r['config'] in ['soft_dpo','oracle_soft_dpo']]
    archive=json.loads((REPO/'artifacts/v7_dev_grid_20261110/RAW_RESULTS_MANIFEST.json').read_text())
    records += [{'role':'previous_evaluation','path':str(OLD/r['path']),'bytes':r['bytes'],'sha256':r['sha256']} for r in archive['files'] if r['path'].startswith('eval/') and r['path'].endswith(('/generations.jsonl','/scored.jsonl'))]
    def check(r):
        p=Path(r['path'])
        assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],r['path']
        return {**r,'verified':True}
    with ThreadPoolExecutor(max_workers=4) as ex:
        checked=list(ex.map(check,records))
    result={'status':'passed','verified_utc':datetime.now(timezone.utc).isoformat(),'files':checked,'note':'Initial policy/RM/tokenizers, exact training/evaluation source, 10 reused baseline exports and all 34 generated/scored files match archived SHA256.'}
    (ROOT/'audit/frozen_reuse_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Frozen reuse audit passed:',len(checked),'files',flush=True)
if __name__=='__main__':main()
