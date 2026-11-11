#!/usr/bin/env python3
import json, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
ROOT=Path('/mnt/hezhisheng/su/experiment_v7_20261110'); SRC=Path('/mnt/hezhisheng/distributionally_robust_dpo-2'); PY=SRC/'.venv/bin/python'
SCORE=Path('/mnt/hezhisheng/su/experiment_handoff_v5/score_skywork.py'); RM=Path('/mnt/hezhisheng/su/experiment_handoff_v5/models/Skywork-Reward-V2-Llama-3.1-8B'); LOGS=ROOT/'logs'
def score(c,gpu):
 n=c['name']; inp=ROOT/'eval'/n/'generations.jsonl'; out=ROOT/'eval'/n/'scored.jsonl'; log=LOGS/f'score_{n}.log'
 if out.exists() and sum(1 for _ in out.open())>=5000:return True
 cmd=[str(PY),str(SCORE),'--model',str(RM),'--input',str(inp),'--output',str(out),'--mode','responses','--batch-size','2','--max-length','4096','--device','cuda:0']
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['PYTHONUNBUFFERED']='1'
 with log.open('w') as f:r=subprocess.run(cmd,cwd=str(SCORE.parent),env=env,stdout=f,stderr=subprocess.STDOUT)
 return r.returncode==0 and out.exists() and sum(1 for _ in out.open())>=5000
def main():
 while Path('/proc/1178105').exists(): time.sleep(60)
 m=json.load(open(ROOT/'v7_grid_manifest.json')); jobs=[]
 for c in m['configs']:
  g=ROOT/'eval'/c['name']/'generations.jsonl'; s=ROOT/'eval'/c['name']/'scored.jsonl'
  if g.exists() and sum(1 for _ in g.open())>=5000 and not(s.exists() and sum(1 for _ in s.open())>=5000):jobs.append(c)
 with ThreadPoolExecutor(max_workers=2) as ex:
  fs=[ex.submit(score,c,str([1,4][i%2])) for i,c in enumerate(jobs)]
  for c,f in zip(jobs,fs):print(c['name'],f.result(),flush=True)
 oracle=ROOT/'eval/oracle_soft_dpo/scored.jsonl'; specs=[]
 for c in m['configs']:
  s=ROOT/'eval'/c['name']/'scored.jsonl'
  if c['name']!='oracle_soft_dpo' and s.exists() and sum(1 for _ in s.open())>=5000:specs.append(f"{c['name']}={s}")
 if oracle.exists() and len(specs)==33:
  cmd=[str(PY),str(Path('/mnt/hezhisheng/su/1030-lastest/experiment_handoff_v5/metrics_shortfall.py')),'--oracle',str(oracle),'--input',*specs,'--output',str(ROOT/'eval/v7_dev_metrics.json'),'--bootstrap','2000','--seed','601']
  with (LOGS/'metrics_v7.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
if __name__=='__main__':main()
