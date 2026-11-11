#!/usr/bin/env python3
"""Descriptive summaries of frozen votes, confidence weights and interval widths."""
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(p):return [json.loads(s) for s in p.open() if s.strip()]
def write(p,rs):
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
m=json.loads((ROOT/'v7_grid_manifest.json').read_text());out=ROOT/'reports';out.mkdir(exist_ok=True)
patterns=[];distributions=[]
for eps in [.15,.2]:
    c=next(c for c in m['configs'] if c['distance']=='kl' and c['rho']==.005 and c['epsilon']==eps)
    rows=read(ROOT/f"data/methods/{c['name']}/train.jsonl")
    for n in [1,3,5]:
        for k in range(n+1):
            r=[r for r in rows if r['n_votes']==n and r['k_1']==k]
            if r:patterns.append({'epsilon':eps,'n_votes':n,'k_1':k,'pairs':len(r),'p_hat':k/n,'unanimous':k in [0,n],'lambda':r[0]['lambda_i']})
    for group in ['all','N=1','N=3','N=5','unanimous','disagreement']:
        rs=[r for r in rows if group=='all' or group==f"N={r['n_votes']}" or (group=='unanimous' and r['k_1'] in [0,r['n_votes']]) or (group=='disagreement' and r['k_1'] not in [0,r['n_votes']])]
        a=np.array([r['lambda_i'] for r in rs])
        distributions.append({'epsilon':eps,'group':group,'pairs':len(rs),'lambda_mean':float(a.mean()),'lambda_min':float(a.min()),'lambda_median':float(np.median(a)),'lambda_max':float(a.max())})
widths=[]
for c in m['configs']:
    if c['distance'] is None:continue
    rs=read(ROOT/f"data/methods/{c['name']}/train.jsonl")
    raw=np.array([r['upper']-r['lower'] for r in rs]);effective=raw if c['method']=='label_dro_common' else raw*np.array([1-r['lambda_i'] for r in rs])
    widths.append({'config':c['name'],'distance':c['distance'],'rho':c['rho'],'epsilon':c['epsilon'],'raw_interval_width_mean':float(raw.mean()),'effective_interval_width_mean':float(effective.mean()),'boundary_label_pairs':sum(r['p_hat'] in [0,1] for r in rs),'pairs':len(rs)})
write(out/'posterior_vote_patterns.csv',patterns);write(out/'lambda_distributions.csv',distributions);write(out/'effective_interval_widths.csv',widths)
print('posterior vote patterns, weight distributions and effective intervals saved')
