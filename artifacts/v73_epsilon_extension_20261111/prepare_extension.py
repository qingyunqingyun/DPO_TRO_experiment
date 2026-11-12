#!/usr/bin/env python3
"""Prepare only protocol 6.5's 16 runs from the actual frozen v7.2 votes."""
import hashlib
import importlib.util
import json
import math
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from datasets import Dataset, DatasetDict
from scipy.special import betainc

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'experiment_v7_20261110'
COMMIT = '33136d47511f477cc0189a82c1e25ae98e805a71'
EPS = [0.15, 0.2]

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def read(p): return [json.loads(s) for s in p.open() if s.strip()]
def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def posterior_trust(n, k, epsilon, a0=.5, b0=.5):
    if int(n) != n or int(k) != k or n < 1 or not 0 <= k <= n:
        raise ValueError('Require integer n>=1 and 0<=k<=n')
    if not 0 <= epsilon <= .2 or not (a0 > 0 and b0 > 0):
        raise ValueError('Invalid tolerance or prior')
    h=k/n
    return float(max(0.,min(1.,betainc(a0+k,b0+n-k,min(1.,h+epsilon))-betainc(a0+k,b0+n-k,max(0.,h-epsilon)))))

def main():
    spec=importlib.util.spec_from_file_location('v72_prepare',ROOT/'audit/v72_prepare_v7_grid.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    helper.posterior_trust=posterior_trust
    base=json.loads((PREVIOUS/'v7_grid_manifest.json').read_text())
    checks=json.loads((PREVIOUS/'audit/input_and_math_checks.json').read_text())
    for rel,expected in checks['input_hashes'].items():
        source=PREVIOUS/rel;assert sha(source)==expected,rel
        target=ROOT/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    pairs={s:read(ROOT/f'data/pairs/{s}.jsonl') for s in ['train','dev']}
    obs={s:read(ROOT/f'data/observations/base_{s}.jsonl') for s in pairs}
    assert Counter(x['n_votes'] for x in obs['train'])=={1:800,3:600,5:600}
    assert sum(x['n_votes'] for x in obs['train'])==5600
    assert not set(x['prompt_id'] for x in pairs['train'])&set(x['prompt_id'] for x in pairs['dev'])
    references={(1,1):[.480502,.549815],(3,2):[.466610,.607254],(3,3):[.696909,.772547],(5,3):[.540727,.686868],(5,4):[.668501,.815393],(5,5):[.808946,.874533]}
    for (n,k),values in references.items():
        for e,v in zip(EPS,values):assert abs(posterior_trust(n,k,e)-v)<1e-6
    for n in [1,3,5]:
        for k in range(n+1):
            weights=[posterior_trust(n,k,e) for e in [0,.025,.05,.1,.15,.2]]
            assert weights==sorted(weights) and weights[0]==0
            for e in EPS:assert abs(posterior_trust(n,k,e)-posterior_trust(n,n-k,e))<1e-12
    for bad in [-.01,.20001,float('nan')]:
        try:posterior_trust(1,1,bad)
        except ValueError:pass
        else:raise AssertionError('Invalid epsilon accepted')
    configs=[]
    for distance,rhos in [('kl',base['kl_radii']),('w1',base['w1_radii'])]:
        for rho in rhos:
            for epsilon in EPS:
                name=f'{distance}_tro_r{rho:g}_e{epsilon:g}'.replace('.','')
                cfg={'name':name,'method':'tro_trust_common','distance':distance,'rho':rho,'epsilon':epsilon}
                original=next(c for c in base['configs'] if c['distance']==distance and c['rho']==rho and c['epsilon']==.1)
                recs={}
                for split in pairs:
                    rows=helper.method_rows(pairs[split],obs[split],method=cfg['method'],distance=distance,rho=rho,epsilon=epsilon)
                    old_rows=read(PREVIOUS/f"data/methods/{original['name']}/{split}.jsonl")
                    assert len(rows)==len(old_rows)
                    for row,old in zip(rows,old_rows):
                        assert {k:v for k,v in row.items() if k not in ['lambda_i','epsilon']}=={k:v for k,v in old.items() if k not in ['lambda_i','epsilon']}
                        assert row['p_star']==.5 and row['lambda_i']>=old['lambda_i']
                    helper.write(ROOT/f'data/methods/{name}/{split}.jsonl',rows);recs[split]=rows
                ds=ROOT/'datasets'/name
                assert not ds.exists(),f'Existing dataset, refuse overwrite: {ds}'
                DatasetDict({s:Dataset.from_list(rows) for s,rows in recs.items()}).save_to_disk(str(ds))
                configs.append({**cfg,'dataset':str(ds),'train_rows':2000,'dev_rows':500,'source_batch':'v73_epsilon_extension','protocol_commit':COMMIT})
    assert len(configs)==16
    registered=(ROOT/'audit/EXPERIMENT_PROTOCOL.md').read_text()
    assert all(c['name'] in registered for c in configs)
    manifest={**base,'protocol':'2026-11-11-v7.3-epsilon-extension','protocol_repo_commit':COMMIT,'source_root':str(PREVIOUS),'epsilon_grid':EPS,'configs':configs,'checks':{'config_count':16,'frozen_inputs_match':True,'reference_table':'passed','symmetry_and_monotonicity':'passed','invalid_epsilon_rejected':'passed','only_epsilon_and_lambda_changed':'passed'}}
    save(ROOT/'v7_grid_manifest.json',manifest)
    save(ROOT/'audit/input_and_math_checks.json',{**checks,'extension_checks':manifest['checks'],'configs':16})
    metadata=json.loads((PREVIOUS/'run_metadata.json').read_text())
    metadata.update({'protocol_repo_commit':COMMIT,'protocol_version':manifest['protocol'],'status':'prepared','prepared_utc':datetime.now(timezone.utc).isoformat(),'previous_batch':str(PREVIOUS),'configs':16,'complete_configs':0,'scored_generations':0,'planned_scored_generations':80000,'finished_scoring_utc':None,'timing':None,'extension_epsilon':EPS,'training_behavior':'Unmodified archived trainer, 31 actual updates, scheduler planned 30; final 32 pair gradients uncommitted, to be verified in run logs.'})
    metadata.pop('started_utc',None)
    save(ROOT/'run_metadata.json',metadata)
    trainer=(PREVIOUS/'audit_trainer.py').read_text().replace("'configs':34","'configs':len(m['configs'])")
    (ROOT/'audit_trainer.py').write_text(trainer)
    save(ROOT/'audit/preparation_hashes.json',[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(ROOT.glob('data/*/*.jsonl'))])
    print('Prepared and audited 16 configs from unchanged frozen votes.',flush=True)

if __name__=='__main__':main()
