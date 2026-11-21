import hashlib, json, os
from pathlib import Path
ROOT=Path('/mnt/hezhisheng/su/oracle_full_20261120')
files=[
 'data/oracle_train_full.jsonl','data/build_manifest.json',
 'datasets/full_oracle/dataset_dict.json','outputs/full_oracle/final_hf/model.safetensors',
 'eval/full_test_generations.jsonl','eval/full_test_scored.jsonl',
 'report/FULL_ORACLE_REPORT.md','report/full_oracle_metrics.json','report/full_oracle_prompt_means.csv',
]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def lines(p):
 try:return sum(1 for _ in p.open())
 except:return None
manifest={
 'run_id':'oracle_full_20261120',
 'status':'complete',
 'interpretation':'all 61005 length-eligible training pairs; all 1995 eligible test prompts held out',
 'data':{'train_pairs':61005,'test_prompts':1995,'train_source':'/mnt/hezhisheng/su/1030-lastest/experiment_handoff_v5/data/raw/train_filtered.jsonl','test_source':'/mnt/hezhisheng/su/1030-lastest/experiment_handoff_v5/data/raw/test_filtered.jsonl'},
 'model':{'policy':'Llama-3.2-1B-Instruct','rm':'Skywork/Skywork-Reward-V2-Llama-3.1-8B','oracle_temperature':9.055908203125,'epochs':2,'max_len':2048,'effective_batch':128,'lr':5e-7,'beta':0.01},
 'evaluation':{'responses_per_prompt':10,'rows':19950,'seed':401,'temperature':0.7,'top_p':0.9,'max_new_tokens':1024,'context':4096,'rm_max_length':4096,'mean_R':7.1573615810327365},
 'artifacts':{},
}
for rel in files:
 p=ROOT/rel
 manifest['artifacts'][rel]={'exists':p.exists(),'bytes':p.stat().st_size if p.exists() else None,'sha256':sha(p) if p.exists() else None,'lines':lines(p) if p.suffix in {'.jsonl','.csv','.md'} and p.exists() else None}
(ROOT/'report/FULL_ORACLE_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(manifest,indent=2,ensure_ascii=False))
