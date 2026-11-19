import json, hashlib, re, os
from pathlib import Path
import random
ROOT=Path('/mnt/hezhisheng/su/1030-lastest/experiment_handoff_v5')
OUT=Path('/mnt/hezhisheng/su/convergence_eval_20261119/audit')
Ns=[1,5,11,51,101]; reps=[(1,201,301,401),(2,202,302,402),(3,203,303,403)]
def rows(p): return [json.loads(x) for x in Path(p).open() if x.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def derived(seed,pair_id,tag): return int.from_bytes(hashlib.sha256(f'{seed}:{tag}:{pair_id}'.encode()).digest()[:8],'big')
def finite(x): return isinstance(x,(int,float)) and x==x
issues=[]; checks=[]; groups=[]
pairs=rows(ROOT/'data/pairs/train.jsonl'); oracle=rows(ROOT/'data/oracle/train.jsonl')
pmap={x['pair_id']:x for x in pairs}; omap={x['pair_id']:x for x in oracle}
base_pair_hash=sha(ROOT/'data/pairs/train.jsonl'); base_oracle_hash=sha(ROOT/'data/oracle/train.jsonl')
# frozen eval candidate checks
eval_sets={}
for split in ['dev','test_diagnostic']:
 p=ROOT/'data/pairs'/f'{split}.jsonl'; rr=rows(p); eval_sets[split]={'path':str(p),'rows':len(rr),'prompt_ids':sha(p),'unique_prompt_ids':len({x['prompt_id'] for x in rr})}
# checkpoint and logs
for rep,vote,trainseed,genseed in reps:
 repaudit={'repeat':rep,'vote_seed':vote,'training_seed':trainseed,'generation_seed':genseed,'soft':{},'oracle':{},'nested_prefix_ok':True}
 expected={}
 # pstar and observations/data audit
 for n in Ns:
  tag=f'conv_r{rep}_n{n}'
  obsp=ROOT/'data/observations'/tag/'train.jsonl'; softp=ROOT/'data/methods'/tag/'soft_dpo_train.jsonl'; oraclep=ROOT/'data/methods'/tag/'oracle_soft_dpo_train.jsonl'
  o=rows(obsp); s=rows(softp); osr=rows(oraclep)
  obsmap={x['pair_id']:x for x in o}; smap={x['pair_id']:x for x in s}; osmap={x['pair_id']:x for x in osr}
  rec={'n':n,'observation_rows':len(o),'soft_rows':len(s),'oracle_rows':len(osr),'observation_sha256':sha(obsp),'soft_sha256':sha(softp),'oracle_sha256':sha(oraclep),'n_votes_values':sorted({x.get('n_votes') for x in o}),'pstar_mismatches':0,'sequence_mismatches':0,'pair_mismatches':0,'oracle_pstar_mismatches':0}
  if len(o)!=len(pairs) or len(s)!=len(pairs) or len(osr)!=len(pairs): issues.append((tag,'row_count',len(o),len(s),len(osr)))
  for pid,pair in pmap.items():
   x=obsmap.get(pid); sm=smap.get(pid); om=osmap.get(pid)
   if not x or not sm or not om: rec['pair_mismatches']+=1; continue
   if float(omap[pid]['p_star']) != float(sm.get('p_star',omap[pid]['p_star'])) if 'p_star' in sm else False: rec['oracle_pstar_mismatches']+=1
   # For oracle records pstar should match frozen oracle; oracle method record has pstar.
   if 'p_star' not in om or float(om['p_star']) != float(omap[pid]['p_star']): rec['oracle_pstar_mismatches']+=1
   if int(x['n_votes'])!=n or not finite(float(omap[pid]['p_star'])): rec['pstar_mismatches']+=1
   rng=random.Random(derived(vote,pid,'votes')); k=sum(rng.random()<float(omap[pid]['p_star']) for _ in range(n))
   if int(x['k_1'])!=k: rec['sequence_mismatches']+=1
   if float(sm.get('p_hat',-1)) != float(x['p_hat']): rec['pstar_mismatches']+=1
  if rec['pstar_mismatches'] or rec['sequence_mismatches'] or rec['pair_mismatches'] or rec['oracle_pstar_mismatches']: issues.append((tag,'data_mismatch',rec))
  expected[n]=obsmap; repaudit['soft'][str(n)]=rec
 # compare prefixes directly
 for pid in pmap:
  ks=[]
  for n in Ns: ks.append(expected[n][pid]['k_1'])
  # recompute monotonic prefix independently; k count should be nondecreasing and consistency through random draws
  rng=random.Random(derived(vote,pid,'votes')); draws=[int(rng.random()<float(omap[pid]['p_star'])) for _ in range(101)]
  cs=[]
  for n in Ns: cs.append(sum(draws[:n]))
  if ks != cs: repaudit['nested_prefix_ok']=False; issues.append((f'conv_r{rep}', 'nested_prefix', pid,ks,cs))
 # checkpoint/logs
 for n in Ns:
  name=f'conv_r{rep}_n{n}_soft'; hf=ROOT/'outputs'/name/'final_hf'; log=ROOT/'logs'/f'train_{name}.log';
  ok=(hf/'model.safetensors').is_file() and (hf/'config.json').is_file() and any((hf/x).exists() for x in ['tokenizer.json','tokenizer.model']) and 'Process' in log.read_text(errors='ignore') and 'exits successfully' in log.read_text(errors='ignore')
  text=log.read_text(errors='ignore')
  cmd=next((x for x in text.splitlines() if 'train_alignment.py' in x and '--tro_method' in x), '')
  checkspec={k:('OK' if v in cmd else 'MISSING') for k,v in {'--tro_method soft_dpo':'--tro_method soft_dpo','--max_epochs 2':'--max_epochs 2','--learning_rate 5e-7':'--learning_rate=5e-7','--beta 0.01':'--beta 0.01','--max_len 2048':'--max_len 2048','--train_batch_size=128':'--train_batch_size=128','--seed '+str(trainseed):'--seed '+str(trainseed)}.items()}
  repaudit['soft'][str(n)].update({'checkpoint_usable':ok,'checkpoint_path':str(hf),'log_success':('exits successfully' in text),'training_command_checks':checkspec})
  if not ok or any(x!='OK' for x in checkspec.values()): issues.append((name,'checkpoint_or_command',ok,checkspec))
 name=f'conv_r{rep}_oracle'; hf=ROOT/'outputs'/name/'final_hf'; log=ROOT/'logs'/f'train_{name}.log'; text=log.read_text(errors='ignore'); cmd=next((x for x in text.splitlines() if 'train_alignment.py' in x and '--tro_method' in x), '')
 ok=(hf/'model.safetensors').is_file() and (hf/'config.json').is_file() and any((hf/x).exists() for x in ['tokenizer.json','tokenizer.model']) and 'exits successfully' in text
 checkspec={k:('OK' if v in cmd else 'MISSING') for k,v in {'--tro_method oracle_soft_dpo':'--tro_method oracle_soft_dpo','--max_epochs 2':'--max_epochs 2','--learning_rate 5e-7':'--learning_rate=5e-7','--beta 0.01':'--beta 0.01','--max_len 2048':'--max_len 2048','--train_batch_size=128':'--train_batch_size=128','--seed '+str(trainseed):'--seed '+str(trainseed)}.items()}
 repaudit['oracle']={'checkpoint_usable':ok,'checkpoint_path':str(hf),'log_success':('exits successfully' in text),'training_command_checks':checkspec,'source_dataset':str(ROOT/'datasets'/f'conv_r{rep}_n1/oracle_soft_dpo')}
 if not ok or any(x!='OK' for x in checkspec.values()): issues.append((name,'checkpoint_or_command',ok,checkspec))
 groups.append(repaudit)
# all checkpoint base config hashes etc
result={'protocol_target':{'Ns':Ns,'repeats':3,'pairs':len(pairs),'train_pair_sha256':base_pair_hash,'train_oracle_sha256':base_oracle_hash,'evaluation_sets':eval_sets},'groups':groups,'issues':issues,'issue_count':len(issues),'status':'PASS' if not issues else 'FAIL'}
(OUT/'growth_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':result['status'],'issue_count':len(issues),'pairs':len(pairs),'eval_sets':eval_sets},ensure_ascii=False,indent=2))
for x in issues[:20]: print('ISSUE',x)
