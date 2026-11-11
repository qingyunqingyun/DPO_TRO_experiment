#!/usr/bin/env python3
"""Prepare the v7.2 KL/W1 full development grid from frozen v5 pairs/oracle."""
from __future__ import annotations
import argparse, hashlib, json, math, random, shutil, sys
from pathlib import Path
from scipy.special import betainc
from datasets import Dataset, DatasetDict
sys.path.insert(0, '/mnt/hezhisheng/distributionally_robust_dpo-2/src')
from tro_math import kl_interval

BASE=Path('/mnt/hezhisheng/su/1030-lastest/experiment_handoff_v5')
DEFAULT_OUT=Path('/mnt/hezhisheng/su/experiment_v7_20261110')
KL_RADII=[0.005,0.05,0.1,0.3]
W1_RADII=[0.01,0.05,0.15,0.3]
EPS=[0.025,0.05,0.1]


def read(path): return [json.loads(x) for x in Path(path).open() if x.strip()]
def write(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
def stable_seed(seed, *parts):
    raw=(str(seed)+':' + ':'.join(map(str,parts))).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')
def w1_interval(h,rho): return max(0.0,h-rho), min(1.0,h+rho)
def posterior_trust(n,k,epsilon,a0=0.5,b0=0.5):
    h=k/n; a=a0+k; b=b0+n-k
    lo=max(0.,h-epsilon); hi=min(1.,h+epsilon)
    return float(max(0.,min(1.,betainc(a,b,hi)-betainc(a,b,lo))))
def make_obs(pairs, oracle, *, split, allocation_seed, vote_seed):
    # Exact 4:3:3 quotas; dev uses proportional 200/150/150.
    n=len(pairs)
    quotas={1:int(round(n*.4)),3:int(round(n*.3)),5:n-int(round(n*.4))-int(round(n*.3))}
    alloc=[n for n,c in quotas.items() for _ in range(c)]
    rng=random.Random(stable_seed(allocation_seed,split,'allocation')); rng.shuffle(alloc)
    out=[]
    for i,pair in enumerate(pairs):
        nvote=alloc[i]; pstar=float(oracle[pair['pair_id']]['p_star'])
        vr=random.Random(stable_seed(vote_seed,split,pair['pair_id'],'votes'))
        k=sum(vr.random()<pstar for _ in range(nvote)); h=k/nvote
        if 2*k>nvote: hard=1
        elif 2*k<nvote: hard=2
        else: hard=1 if random.Random(stable_seed(vote_seed,split,pair['pair_id'],'tie')).getrandbits(1) else 2
        out.append({'pair_id':pair['pair_id'],'n_votes':nvote,'k_1':k,'p_hat':h,'hard_label':hard,
                    'p_star':pstar,'allocation_seed':allocation_seed,'vote_seed':vote_seed,'split':split})
    assert {n:sum(x['n_votes']==n for x in out) for n in (1,3,5)}==quotas
    return out,quotas

def method_rows(pairs, obs, *, method, distance=None, rho=None, epsilon=None, a0=.5,b0=.5):
    om={x['pair_id']:x for x in obs}; out=[]
    for pair in pairs:
        o=om[pair['pair_id']]; h=o['p_hat']; n=o['n_votes']; k=o['k_1']
        if method=='oracle_soft_dpo':
            lower=upper=h; lam=0.; pstar=o['p_star']
            chosen,rejected=pair['response_1'],pair['response_2']
        elif method=='soft_dpo':
            lower=upper=h; lam=0.; pstar=.5
            chosen,rejected=pair['response_1'],pair['response_2']
        else:
            if distance=='kl': lower,upper=kl_interval(h,rho)
            elif distance=='w1': lower,upper=w1_interval(h,rho)
            else: raise ValueError(distance)
            lam=1.0 if method=='label_dro_common' else posterior_trust(n,k,epsilon,a0,b0)
            pstar=.5; chosen,rejected=pair['response_1'],pair['response_2']
        if method=='hard_dpo': chosen,rejected=(pair['response_1'],pair['response_2']) if o['hard_label']==1 else (pair['response_2'],pair['response_1'])
        out.append({'pair_id':pair['pair_id'],'prompt_id':pair['prompt_id'],'prompt':pair['prompt'],
                    'chosen':chosen,'rejected':rejected,'n_votes':n,'k_1':k,'p_hat':h,
                    'hard_label':o['hard_label'],'lower':lower,'upper':upper,'lambda_i':lam,
                    'p_star':pstar,'distance':distance,'rho':rho,'epsilon':epsilon,'prior_a0':a0,'prior_b0':b0})
    return out

def config_grid():
    out=[{'name':'soft_dpo','method':'soft_dpo','distance':None,'rho':None,'epsilon':None},
         {'name':'oracle_soft_dpo','method':'oracle_soft_dpo','distance':None,'rho':None,'epsilon':None}]
    for dist,rhos in [('kl',KL_RADII),('w1',W1_RADII)]:
        for rho in rhos:
            out.append({'name':f'{dist}_dro_r{rho:g}'.replace('.',''), 'method':'label_dro_common','distance':dist,'rho':rho,'epsilon':None})
            for eps in EPS:
                out.append({'name':f'{dist}_tro_r{rho:g}_e{eps:g}'.replace('.',''), 'method':'tro_trust_common','distance':dist,'rho':rho,'epsilon':eps})
    assert len(out)==34
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default=str(DEFAULT_OUT)); ap.add_argument('--allocation-seed',type=int,default=1731); ap.add_argument('--vote-seed',type=int,default=2101); ap.add_argument('--a0',type=float,default=.5); ap.add_argument('--b0',type=float,default=.5)
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    pairs_root=BASE/'data/pairs'; oracle_root=BASE/'data/oracle'
    pairs={'train':read(pairs_root/'train.jsonl'),'dev':read(pairs_root/'dev.jsonl')}; oracle={s:{x['pair_id']:x for x in read(oracle_root/f'{s}.jsonl')} for s in pairs}
    assert len(pairs['train'])==2000 and len(pairs['dev'])==500
    # Copy frozen manifests, not mutable model/checkpoint directories.
    for s in pairs: write(out/f'data/pairs/{s}.jsonl',pairs[s]); write(out/f'data/oracle/{s}.jsonl',oracle[s].values())
    obs={}; quotas={}
    for s in pairs:
        obs[s],quotas[s]=make_obs(pairs[s],oracle[s],split=s,allocation_seed=args.allocation_seed,vote_seed=args.vote_seed)
        write(out/f'data/observations/base_{s}.jsonl',obs[s])
    configs=config_grid(); config_meta=[]
    for cfg in configs:
        tag=cfg['name']; method=cfg['method']; recs={}
        for s in pairs:
            rr=method_rows(pairs[s],obs[s],method=method,distance=cfg['distance'],rho=cfg['rho'],epsilon=cfg['epsilon'],a0=args.a0,b0=args.b0)
            write(out/f'data/methods/{tag}/{s}.jsonl',rr); recs[s]=rr
        dsdir=out/'datasets'/tag
        if dsdir.exists(): shutil.rmtree(dsdir)
        DatasetDict({'train':Dataset.from_list(recs['train']),'dev':Dataset.from_list(recs['dev'])}).save_to_disk(str(dsdir))
        config_meta.append({**cfg,'dataset':str(dsdir),'train_rows':len(recs['train']),'dev_rows':len(recs['dev'])})
    # Numeric and structural acceptance checks.
    checks={'config_count':len(configs),'train_rows':len(pairs['train']),'dev_rows':len(pairs['dev']),'quotas':quotas,'posterior_checks':[]}
    for n in (1,3,5):
        for k in range(n+1):
            vals=[posterior_trust(n,k,e,args.a0,args.b0) for e in EPS]
            assert all(0<=v<=1 for v in vals); checks['posterior_checks'].append({'n':n,'k':k,'lambda':vals})
    manifest={'protocol':'2026-11-10-v7.2-dev-grid','source_root':str(BASE),'policy':'/mnt/hezhisheng/distributionally_robust_dpo-2/models/Llama1b','rm':'/mnt/hezhisheng/su/experiment_handoff_v5/models/Skywork-Reward-V2-Llama-3.1-8B','allocation_seed':args.allocation_seed,'vote_seed':args.vote_seed,'prior':{'a0':args.a0,'b0':args.b0},'epsilon_grid':EPS,'kl_radii':KL_RADII,'w1_radii':W1_RADII,'quotas':quotas,'configs':config_meta,'checks':checks}
    (out/'v7_grid_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'out':str(out),'configs':len(configs),'quotas':quotas,'train':len(pairs['train']),'dev':len(pairs['dev'])},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
