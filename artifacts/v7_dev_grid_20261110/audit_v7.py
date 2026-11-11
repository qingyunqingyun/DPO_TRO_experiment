#!/usr/bin/env python3
"""Offline checks of the frozen inputs and binary TRO math; no GPU allocation."""
import hashlib, json, math, statistics, sys
from collections import Counter
from pathlib import Path
import torch
from transformers import AutoTokenizer
sys.path.insert(0, '/mnt/hezhisheng/distributionally_robust_dpo-2/src')
from tro_math import kl_interval, bernoulli_kl, tro_dpo_loss
from tro_dataset import TRORewardDataset
from openrlhf.datasets.reward_dataset import preprocess_data

ROOT=Path(__file__).resolve().parent
def read(p): return [json.loads(s) for s in p.open() if s.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    m=json.loads((ROOT/'v7_grid_manifest.json').read_text())
    pairs={s:read(ROOT/f'data/pairs/{s}.jsonl') for s in ['train','dev']}
    obs=read(ROOT/'data/observations/base_train.jsonl')
    oracle=read(ROOT/'data/oracle/train.jsonl')
    assert len(set(r['prompt_id'] for r in pairs['train']))==2000
    assert len(set(r['prompt_id'] for r in pairs['dev']))==500
    assert not (set(r['prompt_id'] for r in pairs['train']) & set(r['prompt_id'] for r in pairs['dev']))
    assert Counter(r['n_votes'] for r in obs)=={1:800,3:600,5:600}
    T=statistics.median(abs(r['score_1']-r['score_2']) for r in oracle if r['score_1']!=r['score_2'])
    assert abs(T-9.2548828125)<1e-12
    assert all(abs(r['p_star']-1/(1+math.exp(-(r['score_1']-r['score_2'])/T)))<1e-12 for r in oracle)
    tok=AutoTokenizer.from_pretrained(m['policy'], local_files_only=True)
    lengths=[]; max_dev_prompt=0
    for split,rs in pairs.items():
        for r in rs:
            prompt,c,reject,_=preprocess_data({'prompt':r['prompt'],'chosen':r['response_1'],'rejected':r['response_2']}, None,'prompt','chosen','rejected',tok.apply_chat_template, True)
            for resp in [c,reject]:
                text=(prompt+resp).rstrip('\n')
                if not text.endswith(tok.eos_token):text+=' '+tok.eos_token
                n=len(tok(text, add_special_tokens=False, truncation=False)['input_ids'])
                assert n<=2048, (split,r['pair_id'],n)
                lengths.append(n)
            if split=='dev':
                prompt=tok.apply_chat_template(r['prompt'], add_generation_prompt=True, tokenize=False)
                max_dev_prompt=max(max_dev_prompt,len(tok(prompt,add_special_tokens=True)['input_ids']))
    assert max_dev_prompt+1024<=4096
    post={}
    for c in m['configs']:
        rs=read(ROOT/f"data/methods/{c['name']}/train.jsonl")
        for index,(o,r) in enumerate(zip(obs,rs)):
            assert r['pair_id']==o['pair_id'] and r['p_hat']==o['p_hat'] and r['n_votes']==o['n_votes'] and r['k_1']==o['k_1']
            assert r['chosen']==pairs['train'][index]['response_1'] and r['rejected']==pairs['train'][index]['response_2']
            if c['method']!='oracle_soft_dpo':assert r['p_star']==.5
            h=r['p_hat'];lo=r['lower'];hi=r['upper'];l=r['lambda_i']
            assert 0<=lo<=h<=hi<=1 and 0<=l<=1
            if c['distance']=='w1':
                assert abs(lo-max(0,h-c['rho']))<1e-12 and abs(hi-min(1,h+c['rho']))<1e-12
            if c['distance']=='kl':
                for end in [lo,hi]:
                    if end not in (0,1):assert abs(bernoulli_kl(h,end)-c['rho'])<1e-10
            if c['method']=='tro_trust_common':
                key=(o['n_votes'],o['k_1'],c['epsilon'])
                if key in post:assert abs(post[key]-l)<1e-12
                post[key]=l
    expected={(1,1):[.200475,.282314,.395819],(3,2):[.079967,.159567,.316162],(3,3):[.315472,.436972,.592916],(5,3):[.095648,.190332,.373012],(5,4):[.119024,.236946,.464404],(5,5):[.394083,.537276,.707482]}
    for (n,k),vals in expected.items():
        for e,v in zip(m['epsilon_grid'],vals):
            assert abs(post[n,k,e]-v)<1e-6
            assert abs(post[n,k,e]-post[n,n-k,e])<1e-12
        assert post[n,k,.025]<=post[n,k,.05]<=post[n,k,.1]
    # Loss and gradients at both sides of zero, zero subgradient, and degeneracies.
    for h in [0.,.2,1/3,.6,1.]:
      for dist in ['kl','w1']:
        for rho in [0.,.005,.05,.3]:
          lo,hi=kl_interval(h,rho) if dist=='kl' else (max(0,h-rho),min(1,h+rho))
          for lam in [0.,.3,1.]:
            for z0 in [-2.,0.,2.]:
              v=torch.tensor([z0],dtype=torch.float32,requires_grad=True); zero=torch.zeros_like(v)
              hv=torch.tensor([h]);lov=torch.tensor([lo]);hiv=torch.tensor([hi]);lv=torch.tensor([lam])
              loss,z,target,worst=tro_dpo_loss(v,zero,zero,zero,1.,hv,lov,hiv,lv)
              loss.sum().backward()
              w=lo if z0>0 else hi if z0<0 else h;t=lam*h+(1-lam)*w
              assert abs(v.grad.item()-(torch.sigmoid(v).item()-t))<2e-7
              assert abs(loss.item()-(torch.nn.functional.softplus(v).item()-t*z0))<5e-7
              assert not target.requires_grad and not worst.requires_grad
    (ROOT/'audit/input_and_math_checks.json').write_text(json.dumps({'status':'passed','oracle_temperature':T,'train_counts':dict(Counter(r['n_votes'] for r in obs)),'train_votes':sum(r['n_votes'] for r in obs),'train_dev_prompt_overlap':0,'max_training_side_tokens':max(lengths),'max_dev_prompt_tokens_including_generation_template':max_dev_prompt,'lambda_reference_table':'passed','loss_and_gradient_checks':'passed','input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in ROOT.glob('data/*/*.jsonl') if 'methods' not in str(p)}},indent=2)+'\n')
    print('Input, length, oracle, metadata isolation, KL/W1, posterior table and loss/gradient audits passed.')
if __name__=='__main__':main()
