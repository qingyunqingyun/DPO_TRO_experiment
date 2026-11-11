#!/usr/bin/env python3
"""Exercise the actual trainer loss dispatch on CPU with cached dataset records."""
import json, sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,'/mnt/hezhisheng/distributionally_robust_dpo-2/src')
from tro_trainer import TROTrainer
ROOT=Path(__file__).resolve().parent
m=json.loads((ROOT/'v7_grid_manifest.json').read_text());checked=0
for c in m['configs']:
    rows=[json.loads(s) for s in (ROOT/f"data/methods/{c['name']}/train.jsonl").open()][:6]
    n=len(rows)
    for sign in [-1,0,1]:
        tr=object.__new__(TROTrainer);tr.tro_method=c['method'];tr.beta=.01;tr.args=SimpleNamespace()
        x=torch.full((n,),float(sign*3),requires_grad=True);z0=torch.zeros(n)
        loss,z,target,worst,_,_=tr._loss(x,z0,z0,z0,rows)
        expected=[]
        for r in rows:
            h=r['p_hat'];w=r['lower'] if sign>0 else r['upper'] if sign<0 else h
            if c['method']=='soft_dpo':t=h
            elif c['method']=='oracle_soft_dpo':t=r['p_star']
            elif c['method']=='label_dro_common':t=w
            else:t=r['lambda_i']*h+(1-r['lambda_i'])*w
            expected.append(t)
        ref=torch.tensor(expected)
        assert torch.allclose(target,ref,atol=1e-7,rtol=1e-6)
        loss.mean().backward();assert torch.allclose(x.grad,.01*(torch.sigmoid(z)-ref)/n,atol=1e-8,rtol=1e-5)
        # Equal pair weighting and accumulation equivalence of the loss branch.
        one=[]
        for i in range(n):
            y=x.detach()[i:i+1].clone().requires_grad_(True);o=torch.zeros(1)
            l,*_=tr._loss(y,o,o,o,[rows[i]])
            (l.mean()/n).backward();one.append(y.grad)
        assert torch.allclose(x.grad,torch.cat(one),atol=1e-8,rtol=1e-5)
        checked+=1
(ROOT/'audit/trainer_loss_checks.json').write_text(json.dumps({'status':'passed','configs':len(m['configs']),'cases':checked,'targets_and_gradients':'passed','pair_weighting_and_microbatch_loss_equivalence':'passed','scope':'Actual trainer loss dispatcher on CPU; full DeepSpeed distributed accumulation covered by existing trainer behavior and run logs, not this unit audit.'},indent=2)+'\n')
print('actual trainer dispatch and microbatch loss audit passed:',checked)
