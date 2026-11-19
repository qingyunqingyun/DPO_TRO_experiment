import json,hashlib,math
from pathlib import Path
from collections import defaultdict
OUT=Path('/mnt/hezhisheng/su/convergence_eval_20261119'); G=OUT/'generations'; S=OUT/'scored'
expected=['conv_r%d_%s'%(r,s) for r in (1,2,3) for s in ['oracle','n1_soft','n5_soft','n11_soft','n51_soft','n101_soft']]
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def check(p, scored=False):
 rows=[json.loads(x) for x in p.open() if x.strip()]; d=defaultdict(list)
 for r in rows: d[r.get('prompt_id')].append(r)
 issues=[]
 if len(rows)!=5000 or len(d)!=500 or sorted({len(v) for v in d.values()})!=[10]: issues.append({'rows':len(rows),'prompts':len(d),'counts':sorted({len(v) for v in d.values()})})
 if scored and any(not math.isfinite(float(x.get('skywork_score',float('nan')))) for x in rows): issues.append('nonfinite score')
 return {'rows':len(rows),'prompts':len(d),'prompt_ids':sorted(d),'sha256':sha(p),'issues':issues,'size_bytes':p.stat().st_size}
res={'expected_models':expected,'generation':{},'scored':{},'issues':[]}
for name in expected:
 for typ,base,sc in [('generation',G,False),('scored',S,True)]:
  p=base/f'{name}.jsonl'
  if not p.exists(): res['issues'].append(f'missing {typ}/{name}'); continue
  x=check(p,sc); res[typ][name]=x
  if x['issues']: res['issues'].append({'file':f'{typ}/{name}','issues':x['issues']})
# all prompt sets equal to frozen test and repeat-matched oracle
frozen={x['prompt_id'] for x in (Path('/mnt/hezhisheng/su/1030-lastest/experiment_handoff_v5/data/pairs/test_diagnostic.jsonl')).open() if x.strip() for x in [json.loads(x)]}
res['frozen_eval_prompt_count']=len(frozen)
for typ in ('generation','scored'):
 for name,x in res[typ].items():
  if set(x['prompt_ids'])!=frozen: res['issues'].append(f'{typ}/{name} prompt mismatch')
res['status']='PASS' if not res['issues'] else 'FAIL'
(OUT/'audit'/'post_audit.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':res['status'],'models':len(expected),'issues':len(res['issues'])},ensure_ascii=False))
