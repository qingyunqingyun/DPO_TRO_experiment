#!/usr/bin/env python3
"""Score fixed pairs or generated responses with Skywork Reward V2."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def render(tok, conv):
    text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
    bos = tok.bos_token
    if bos and text.startswith(bos):
        text = text[len(bos):]
    return text

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True); p.add_argument('--input',required=True); p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['pairs','responses'],required=True)
    p.add_argument('--batch-size',type=int,default=2); p.add_argument('--max-length',type=int,default=16384)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--model-revision',default=None)
    a=p.parse_args()
    rows=[json.loads(x) for x in Path(a.input).open() if x.strip()]
    tok=AutoTokenizer.from_pretrained(a.model,use_fast=True,trust_remote_code=True)
    model=AutoModelForSequenceClassification.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa',num_labels=1).to(a.device).eval()
    def score(convs):
        texts=[render(tok,c) for c in convs]
        enc=tok(texts,padding=True,truncation=True,max_length=a.max_length,return_tensors='pt')
        enc={k:v.to(a.device) for k,v in enc.items()}
        with torch.inference_mode():
            val=model(**enc).logits
        if val.ndim==2: val=val[:,0]
        return val.float().cpu().tolist(), enc['attention_mask'].sum(1).cpu().tolist()
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as f:
        for start in range(0,len(rows),a.batch_size):
            batch=rows[start:start+a.batch_size]
            if a.mode=='pairs':
                conv=[]
                for r in batch: conv += [r['prompt']+r['response_1'], r['prompt']+r['response_2']]
                vals,lens=score(conv)
                for i,r in enumerate(batch):
                    x=dict(r); x.update(score_1=float(vals[2*i]),score_2=float(vals[2*i+1]),teacher_model=a.model,teacher_revision=a.model_revision,score_field='logits_scalar',max_length=a.max_length,token_length_1=int(lens[2*i]),token_length_2=int(lens[2*i+1])); f.write(json.dumps(x,ensure_ascii=False)+'\n')
            else:
                vals,lens=score([r['prompt']+r['response'] for r in batch])
                for r,v,l in zip(batch,vals,lens):
                    x=dict(r); x['skywork_score']=float(v); x['token_length']=int(l); f.write(json.dumps(x,ensure_ascii=False)+'\n')
    print(json.dumps({'rows':len(rows),'output':str(out),'model':a.model,'mode':a.mode},ensure_ascii=False))
if __name__=='__main__': main()
