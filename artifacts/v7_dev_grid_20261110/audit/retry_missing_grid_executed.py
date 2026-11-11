#!/usr/bin/env python3
import json, os, subprocess, time
from pathlib import Path
ROOT=Path('/mnt/hezhisheng/su/experiment_v7_20261110'); SRC=Path('/mnt/hezhisheng/distributionally_robust_dpo-2'); PY=SRC/'.venv/bin/python'
def model_ok(name):
 for hf in [ROOT/'outputs'/name/'final_hf', ROOT/'outputs'/name/'checkpoints_dpo'/'final_hf']:
  if (hf/'config.json').exists() and any(hf.glob('*.safetensors')): return True
 return False
def main():
 while True:
  st=ROOT/'train_status.json'
  if st.exists() and len(json.load(open(st)))>=34: break
  time.sleep(60)
 m=json.load(open(ROOT/'v7_grid_manifest.json'))
 missing=[c for c in m['configs'] if not model_ok(c['name'])]
 (ROOT/'retry_plan.json').write_text(json.dumps({'missing':[c['name'] for c in missing]},indent=2))
 for c in missing:
  name=c['name']; out=ROOT/'outputs'/name; log=ROOT/'logs'/f'retry_{name}.log'; out.mkdir(parents=True,exist_ok=True)
  cmd=[str(PY),'-m','deepspeed.launcher.runner','--include','localhost:1,3','--master_port','29991',str(SRC/'src/train_alignment.py'),
   '--save_path',str(out),'--ckpt_path',str(out/'checkpoints_dpo'),'--save_hf_ckpt','--disable_ds_ckpt','--save_steps=-1','--eval_steps=-1','--micro_train_batch_size=1','--train_batch_size=128','--gradient_checkpointing','--zero_stage=2','--bf16','--no-flash_attn','--torch_adam','--learning_rate=5e-7','--lr_warmup_ratio=0.1','--max_norm=1.0','--adam_betas','0.9','0.95','--l2=0','--zpg=2','--train_task=tro',f"--tro_method={c['method']}",'--max_epochs=2','--beta=0.01','--pretrain',str(SRC/'models/Llama1b'),'--ref_pretrain',str(SRC/'models/Llama1b'),'--dataset',c['dataset'],'--prompt_key=prompt','--chosen_key=chosen','--rejected_key=rejected','--apply_chat_template','--max_len=2048','--dataset_num_processors=1','--train_split=train','--eval_split=dev','--seed=1101']
  env=os.environ.copy(); env.update({'DS_SKIP_CUDA_CHECK':'1','VLLM_CONFIGURE_LOGGING':'0','PYTHONUNBUFFERED':'1'})
  with log.open('w') as f:
   f.write(' '.join(cmd)+'\n'); f.flush(); r=subprocess.run(cmd,cwd=str(SRC),env=env,stdout=f,stderr=subprocess.STDOUT)
  print(name,r.returncode,'ok='+str(model_ok(name)),flush=True)
 print('retry complete',flush=True)
if __name__=='__main__': main()
