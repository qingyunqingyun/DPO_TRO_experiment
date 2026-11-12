#!/usr/bin/env python3
"""Validate the extension and export 16/50 results and 32/80 comparisons."""
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from safetensors import safe_open

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'experiment_v7_20261110'
REPO=Path('/mnt/hezhisheng/DPO_TRO_experiment')
ARTIFACT=REPO/'artifacts/v73_epsilon_extension_20261111'
INITIAL_COMMIT='7beb6b46328c33731ac5684dd5939fc1463fd4fe'
EXTENSION_COMMIT='33136d47511f477cc0189a82c1e25ae98e805a71'

def read(p):return [json.loads(s) for s in p.open() if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def csvout(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def utc(ts):return datetime.fromtimestamp(ts,timezone.utc).isoformat()

def main():
    manifest=json.loads((ROOT/'v7_grid_manifest.json').read_text())
    previous=json.loads((OLD/'v7_grid_manifest.json').read_text())
    new=manifest['configs'];configs=previous['configs']+new
    assert len(new)==16 and len(configs)==50 and len(set(c['name'] for c in configs))==50
    new_names={c['name'] for c in new}
    old_artifact=json.loads((REPO/'artifacts/v7_dev_grid_20261110/ARTIFACT_MANIFEST.json').read_text())
    old_payload={x['path']:x for x in old_artifact['payload_files']}
    cached=OLD/'reports/prompt_scores.csv'
    assert sha(cached)==old_payload['reports/prompt_scores.csv']['sha256']
    old_scored=json.loads((OLD/'audit/eval_checks.json').read_text())['score_files']
    # Recheck scores when finalizing, so changed files cannot silently enter reuse.
    for name,record in old_scored.items():assert sha(Path(record['path']))==record['sha256'],name
    expected={r['prompt_id']:r for r in read(ROOT/'data/pairs/dev.jsonl')}
    keys=sorted(expected);assert len(keys)==500
    vectors={};shortages={};prompt_rows=[];lengths=[];score_hashes={}
    cached_rows=list(csv.DictReader(cached.open()))
    for cfg in previous['configs']:
        rs={r['prompt_id']:r for r in cached_rows if r['config']==cfg['name']}
        assert set(rs)==set(keys)
        vectors[cfg['name']]=np.array([float(rs[k]['mean_reward']) for k in keys])
    tok=AutoTokenizer.from_pretrained(manifest['rm'],local_files_only=True)
    for cfg in new:
        name=cfg['name'];p=ROOT/f'eval/{name}/scored.jsonl';g=ROOT/f'eval/{name}/generations.jsonl'
        rows=read(p);gens=read(g);assert len(rows)==len(gens)==5000
        groups=defaultdict(list);seen=set();actual=[];texts=[]
        for row,generation in zip(rows,gens):
            pid=row['prompt_id'];sample=row['sample_id']
            assert pid in expected and row['pair_id']==expected[pid]['pair_id']
            assert row['prompt']==expected[pid]['prompt'] and row['seed']==2101
            assert isinstance(sample,int) and 0<=sample<10 and (pid,sample) not in seen
            seen.add((pid,sample));assert all(row[k]==v for k,v in generation.items())
            value=float(row['skywork_score']);assert math.isfinite(value);groups[pid].append(value)
            text=tok.apply_chat_template(row['prompt']+row['response'],tokenize=False,add_generation_prompt=False)
            if tok.bos_token and text.startswith(tok.bos_token):text=text[len(tok.bos_token):]
            texts.append(text)
        for i in range(0,5000,128):actual.extend(map(len,tok(texts[i:i+128],padding=False,truncation=False)['input_ids']))
        assert max(actual)<=4096 and all(int(r['token_length'])==n for r,n in zip(rows,actual)),name
        assert set(groups)==set(keys) and all(len(values)==10 for values in groups.values())
        vectors[name]=np.array([np.mean(groups[k]) for k in keys])
        lengths.append({'config':name,'samples':5000,'full_rm_max_tokens':max(actual),'rm_truncated_samples':0})
        score_hashes[name]={'scored':{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)},'generations':{'path':str(g),'bytes':g.stat().st_size,'sha256':sha(g)}}
        print('Validated full responses and RM input:',name,flush=True)
    oracle=vectors['oracle_soft_dpo']
    shortages={name:np.maximum(oracle-values,0) for name,values in vectors.items()}
    idx=np.random.default_rng(601).integers(0,500,size=(2000,500))
    def ci(v):return list(map(float,np.quantile(v[idx].mean(axis=1),[.025,.975])))
    results=[]
    for cfg in configs:
        name=cfg['name'];v=vectors[name];s=shortages[name];rci=ci(v);sci=ci(s)
        extension=name in new_names;batch='v73_epsilon_extension_20261111' if extension else 'v7_dev_grid_20261110'
        source=ROOT if extension else OLD
        results.append({'config':name,'method':cfg['method'],'distance':cfg['distance'] or '', 'rho':cfg['rho'],'epsilon':cfg['epsilon'],
            'prompts':500,'samples':5000,'mean_reward':float(v.mean()),'reward_ci95_low':rci[0],'reward_ci95_high':rci[1],
            'oracle_shortfall':float(s.mean()),'shortfall_ci95_low':sci[0],'shortfall_ci95_high':sci[1],'gap_to_oracle':float((v-oracle).mean()),
            'source_batch':batch,'protocol_commit':EXTENSION_COMMIT if extension else INITIAL_COMMIT,
            'train_seed':1101,'allocation_seed':1731,'vote_seed':2101,'generation_seed':2101,'training_seed_repeats':1,
            'epochs':2,'max_len':2048,'learning_rate':5e-7,'dpo_beta':.01,'prior_a0':.5,'prior_b0':.5,'effective_batch':128,'dp_world_size':2,'microbatch':1,'accumulation':64,
            'generation_temperature':.7,'generation_top_p':.9,'max_new_tokens':1024,'generation_context':4096,
            'scored_path':str(source/f'eval/{name}/scored.jsonl'),'checkpoint_path':str(source/f'outputs/{name}/checkpoints_dpo/final_hf'),'training_log':str(source/f'logs/train_{name}.log')})
        for k,r,short,o in zip(keys,v,s,oracle):prompt_rows.append({'config':name,'prompt_id':k,'mean_reward':float(r),'oracle_mean_reward':float(o),'positive_shortfall':float(short),'source_batch':batch})
    old_results={r['config']:r for r in csv.DictReader((OLD/'reports/dev_grid_results.csv').open())}
    for row in results:
        if row['config'] not in new_names:
            for k in ['mean_reward','oracle_shortfall','reward_ci95_low','reward_ci95_high','shortfall_ci95_low','shortfall_ci95_high']:
                assert abs(row[k]-float(old_results[row['config']][k]))<1e-12,(row['config'],k)
    comparisons=[]
    for cfg in configs:
        if cfg['method']!='tro_trust_common':continue
        n=cfg['name'];dro=next(c['name'] for c in configs if c['method']=='label_dro_common' and c['distance']==cfg['distance'] and c['rho']==cfg['rho'])
        for baseline in [dro,'soft_dpo']:
            dr=vectors[n]-vectors[baseline];ds=shortages[n]-shortages[baseline];rci=ci(dr);sci=ci(ds)
            comparisons.append({'config':n,'distance':cfg['distance'],'rho':cfg['rho'],'epsilon':cfg['epsilon'],'baseline':baseline,
                'delta_reward':float(dr.mean()),'delta_reward_ci95_low':rci[0],'delta_reward_ci95_high':rci[1],
                'delta_shortfall':float(ds.mean()),'delta_shortfall_ci95_low':sci[0],'delta_shortfall_ci95_high':sci[1],
                'source_batch':'v73_epsilon_extension_20261111' if n in new_names else 'v7_dev_grid_20261110','protocol_commit':EXTENSION_COMMIT if n in new_names else INITIAL_COMMIT})
    ext_results=[r for r in results if r['config'] in new_names];ext_comps=[r for r in comparisons if r['config'] in new_names]
    assert (len(ext_results),len(results),len(ext_comps),len(comparisons))==(16,50,32,80)
    report=ROOT/'reports'
    for filename,rows in [('extension_results.csv',ext_results),('combined_results.csv',results),('extension_comparisons.csv',ext_comps),('combined_comparisons.csv',comparisons),('extension_prompt_scores.csv',[r for r in prompt_rows if r['config'] in new_names]),('combined_prompt_scores.csv',prompt_rows),('length_audit.csv',lengths)]:csvout(report/filename,rows)
    save(ROOT/'eval/combined_metrics.json',{'configs':50,'new_configs':16,'new_samples':80000,'combined_samples':250000,'bootstrap_replicates':2000,'bootstrap_seed':601,'confidence_interval_unit':'prompt','results':results,'comparisons':comparisons})
    exports=[];training=[]
    for cfg in new:
        name=cfg['name'];out=ROOT/'outputs'/name;hf=out/'checkpoints_dpo/final_hf';log=ROOT/f'logs/train_{name}.log'
        text=log.read_text(errors='replace')
        assert 'exits successfully' in text and re.search(r'Train epoch: 100%.*?2/2',text),name
        for epoch in [0,1]:assert re.search(rf'Train step of epoch {epoch}: 100%.*?1000/1000',text),name
        assert re.search(r'gradient_accumulation_steps\s+\.*\s*64',text),name
        assert re.search(r'train_micro_batch_size_per_gpu\s+1',text),name
        assert 'dist_world_size=2' in text,name
        status=json.loads((out/'status.json').read_text());assert status['returncode']==0
        for p in hf.glob('*.safetensors'):
            with safe_open(str(p),framework='pt',device='cpu') as weights:count=len(weights.keys())
            assert count>0
            exports.append({'config':name,'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'tensor_count':count})
        assert exports[-1]['config']==name
        training.append({'config':name,'log':str(log),'log_sha256':sha(log),'epochs':2,'microsteps_per_rank':2000,'dp_world_size':2,'microbatch':1,'accumulation':64,'optimizer_updates':2000//64,'scheduler_planned_updates':2*(1000//64),'final_uncommitted_pairs':(2000%64)*2,'evidence':'Both epochs completed 1000 steps; DP/accumulation verified from DeepSpeed log; counts derived from unchanged archived trainer/DeepSpeed boundary rule.'})
        print('Verified model export and training log:',name,flush=True)
    assert len({x['sha256'] for x in exports})==16
    save(ROOT/'audit/model_exports.json',exports);save(ROOT/'audit/training_budget_checks.json',{'status':'passed','configs':training})
    save(ROOT/'audit/eval_checks.json',{'status':'passed','configs':16,'total_samples':80000,'all_rewards_finite':True,'all_sample_ids_unique':True,'rm_truncated_samples':0,'max_rm_tokens':max(r['full_rm_max_tokens'] for r in lengths),'files':score_hashes,'old_metrics_reproduced':True})
    states=[json.loads((ROOT/'outputs'/c['name']/'status.json').read_text()) for c in new]
    save(ROOT/'audit/implementation_hashes.json',[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(ROOT.glob('*.py'))+sorted((ROOT/'audit').glob('*.py'))])
    timing={'started_utc':utc(min(s['start_time'] for s in states)),'last_model_utc':utc(max((ROOT/f"outputs/{c['name']}/checkpoints_dpo/final_hf/model.safetensors").stat().st_mtime for c in new)), 'last_scoring_utc':utc(max((ROOT/f"eval/{c['name']}/scored.jsonl").stat().st_mtime for c in new))}
    timing['wall_hours']=(max((ROOT/f"eval/{c['name']}/scored.jsonl").stat().st_mtime for c in new)-min(s['start_time'] for s in states))/3600
    save(ROOT/'audit/timing.json',timing)
    meta=json.loads((ROOT/'run_metadata.json').read_text());meta.update(status='completed',complete_configs=16,scored_generations=80000,finished_scoring_utc=timing['last_scoring_utc'],timing=timing)
    save(ROOT/'run_metadata.json',meta)
    # Reuse and extend the exact descriptive posterior/interval diagnostics.
    diag=(OLD/'diagnostics_v7.py').read_text().replace('for eps in [.025,.05,.1]:','for eps in [.15,.2]:')
    (ROOT/'diagnostics_extension.py').write_text(diag)
    subprocess.run([sys.executable,str(ROOT/'diagnostics_extension.py')],check=True)
    plot=(OLD/'plot_v7.py').read_text().replace("reports/dev_grid_results.csv","reports/combined_results.csv").replace('(4,4)','(4,6)').replace('[None,.025,.05,.1]','[None,.025,.05,.1,.15,.2]').replace("range(4),['DRO','TRO eps=.025','TRO eps=.05','TRO eps=.1']","range(6),['DRO','.025','.05','.1','.15','.2']").replace('for j in range(4):','for j in range(6):').replace('figsize=(13,9)','figsize=(16,9)').replace('v7.2 dev grid:','v7.3 extended dev grid:')
    (ROOT/'plot_extension.py').write_text(plot)
    subprocess.run(['/usr/bin/python',str(ROOT/'plot_extension.py')],check=True)
    best_r=max([r for r in results if r['config']!='oracle_soft_dpo'],key=lambda r:r['mean_reward'])
    best_s=min([r for r in results if r['config']!='oracle_soft_dpo'],key=lambda r:r['oracle_shortfall'])
    lines=['# v7.3 epsilon 扩展实验报告','',f'绑定纲要 commit：`{EXTENSION_COMMIT}`。新增 epsilon=0.15、0.2 的 16 组 TRO 已完成训练、生成、RM 评分与核验，共 80,000 条。复用原 34 组并合并为 50 组、250,000 条。','',
        '本批是在查看首批开发集结果后登记的网格扩展。全部结果为单一 seed、500 道开发题，不属于独立 final test。', '',
        f"合并网格平均奖励最高：`{best_r['config']}`，R={best_r['mean_reward']:.6f}；oracle 短缺最小：`{best_s['config']}`，S_O={best_s['oracle_shortfall']:.6f}。这是开发集描述性极值，不能作为独立显著性证明。",'',
        '## 参数与执行','',
        '复用 Llama-3.2-1B-Instruct 起点与 Skywork-Reward-V2-Llama-3.1-8B。训练对/开发题=2000/500；N=1/3/5 对应800/600/600，共5600票；Beta(0.5,0.5)，T_pref=9.2548828125。allocation/vote/train/generation seed=1731/2101/1101/2101。', '',
        '新增 epsilon=0.15/0.2，各与 KL rho=0.005/0.05/0.1/0.3、W1 rho=0.01/0.05/0.15/0.3 完整组合。每组独立从同一初始策略训练2 epochs；max_len=2048、lr=5e-7、beta=0.01、DP=2、microbatch=1、累积64、有效batch128；AdamW betas=(0.9,0.95)、weight_decay=0、warmup=0.1、clip=1；BF16、FP32 logprob/loss、ZeRO-2、gradient checkpointing、no-flash_attn。', '',
        '每模型每题生成10条，temperature=0.7、top_p=0.9、max_new_tokens=1024、context=4096、vLLM utilization=0.4。RM BF16、responses模式、batch=2、max_length=4096。R为逐题均值的平均；S_O为逐题相对原oracle均值短缺的正部平均。配对bootstrap=2000、seed=601，单位为prompt。', '',
        f"训练到最后评分墙钟约 {timing['wall_hours']:.2f} 小时，包含等待、初始化与重试；不是 GPU 活跃计算时间。UTC 起止见 `../audit/timing.json`。",'',
        '## 新增16组完整结果','', '| 配置 | rho | epsilon | R | R 95% CI | S_O | S_O 95% CI |','|---|---:|---:|---:|---|---:|---|']
    for r in ext_results:lines.append(f"| {r['config']} | {r['rho']} | {r['epsilon']} | {r['mean_reward']:.6f} | [{r['reward_ci95_low']:.6f}, {r['reward_ci95_high']:.6f}] | {r['oracle_shortfall']:.6f} | [{r['shortfall_ci95_low']:.6f}, {r['shortfall_ci95_high']:.6f}] |")
    lines+=['','## 新增配对比较','','ΔR=TRO−baseline，越大越好；ΔS_O=TRO−baseline，越小越好。区间未做多重比较校正。','','| 配置 | baseline | ΔR | ΔR 95% CI | ΔS_O | ΔS_O 95% CI |','|---|---|---:|---|---:|---|']
    for r in ext_comps:lines.append(f"| {r['config']} | {r['baseline']} | {r['delta_reward']:+.6f} | [{r['delta_reward_ci95_low']:+.6f}, {r['delta_reward_ci95_high']:+.6f}] | {r['delta_shortfall']:+.6f} | [{r['delta_shortfall_ci95_low']:+.6f}, {r['delta_shortfall_ci95_high']:+.6f}] |")
    lines+=['','## 新增批次方向统计','','| 距离 | baseline | 配置数 | ΔR>0 | ΔS_O<0 | 两者同时成立 |','|---|---|---:|---:|---:|---:|']
    for distance in ['kl','w1']:
        for base in ['same_radius_dro','soft_dpo']:
            rs=[r for r in ext_comps if r['distance']==distance and (r['baseline']=='soft_dpo')==(base=='soft_dpo')]
            lines.append(f"| {distance} | {base} | {len(rs)} | {sum(r['delta_reward']>0 for r in rs)} | {sum(r['delta_shortfall']<0 for r in rs)} | {sum(r['delta_reward']>0 and r['delta_shortfall']<0 for r in rs)} |")
    lines+=['','## 合并50组结果','','![全部半径与五档epsilon](dev_grid_heatmap.png)','', '| 配置 | 批次 | rho | epsilon | R | S_O |','|---|---|---:|---:|---:|---:|']
    for r in results:lines.append(f"| {r['config']} | {r['source_batch']} | {r['rho']} | {r['epsilon']} | {r['mean_reward']:.6f} | {r['oracle_shortfall']:.6f} |")
    lines+=['','## 核验与限制','',
        '- 已复核初始模型、RM、tokenizer、实际执行源码及复用参照的 SHA256；冻结输入未重新抽票。新增输入仅修改epsilon和lambda。后验数值表、对称、单调、范围拒绝和实际trainer loss/梯度检查通过。',
        '- 新增80000条均具备完整500题×10条，prompt/response与生成文件一致，分数有限；按原RM模板重新计算不截断输入长度，全部不超过4096。',
        '- 新增模型均完成两轮、每rank2000 microstep，实际DP=2、累积64。沿用旧trainer：31个完整更新、scheduler计划30、末尾32对累积梯度未提交。核验依据为实际日志与未改动实现，见training_budget_checks.json。',
        '- 旧34组结果与原归档指标逐项复算一致。旧批次的失败重试、尾批等偏差仍见原报告，未因合并而抹去。新增失败重试保留在本轮日志。',
        '- 本轮只有一个投票/训练seed；置信区间反映prompt间变异，不覆盖重训、重新投票和观察结果后的网格扩展。未启动final test、人数增长或先验敏感性训练。',
        '- 沿用原生成实现，未记录finish_reason，不能精确汇报达到max_new_tokens的比例。', '',
        '## 交付文件','',
        '- extension_results.csv / extension_comparisons.csv：新增16行结果与32行比较。',
        '- combined_results.csv / combined_comparisons.csv：合并50行参数结果与80行比较，注明来源批次及纲要commit。',
        '- extension_prompt_scores.csv / combined_prompt_scores.csv：新增与合并逐题分数。',
        '- length_audit.csv、后验/区间诊断表、PNG/PDF图、../audit/、../run_metadata.json：核验、参数、hash及耗时。',
        f'- 原始评分、生成、模型与日志：服务器 `{ROOT}`；旧批次：`{OLD}`。', '']
    (report/'V73_EPSILON_EXTENSION_REPORT.md').write_text('\n'.join(lines))
    # Publish reviewable tables/source locally; raw weights stay on shared storage.
    ARTIFACT.mkdir(parents=True,exist_ok=True)
    for directory in ['reports','audit']:
        shutil.copytree(ROOT/directory,ARTIFACT/directory,dirs_exist_ok=True)
    for p in list(ROOT.glob('*.py'))+[ROOT/'run_metadata.json',ROOT/'v7_grid_manifest.json',ROOT/'train_status.json',ROOT/'eval_status.json']:
        shutil.copy2(p,ARTIFACT/p.name)
    (ARTIFACT/'README.md').write_text('# v7.3 epsilon extension\n\n[完整报告](reports/V73_EPSILON_EXTENSION_REPORT.md)。新增16组、80000条；合并50组、250000条。所有表含来源批次和纲要commit。\n\n完整生成、评分、训练日志与权重保存在共享服务器 `'+str(ROOT)+'`，访问需共享盘权限，位置和SHA256见audit。旧批次完整结果仍位于`../v7_dev_grid_20261110/`。本次为观察旧开发集结果后登记的扩展，非独立final test。\n')
    save(ARTIFACT/'ARTIFACT_MANIFEST.json',{'source_directory':str(ROOT),'payload_files':[{'path':str(p.relative_to(ARTIFACT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(ARTIFACT.rglob('*')) if p.is_file() and p.name!='ARTIFACT_MANIFEST.json']})
    status_path=REPO/'EXPERIMENT_STATUS.md';text=status_path.read_text()
    text=text.replace('状态为进行中（2026-11-11 15:07 北京时间启动）。','状态为分析完成（2026-11-11 15:07 北京时间启动；结束时间见本轮报告）。')
    text=text.replace('40000 | 进行中，首组训练已启动','40000 | 分析完成').replace('40000 | 已排队','40000 | 分析完成').replace('80000 | 进行中','80000 | 分析完成')
    text+='\n本批已完成16/16训练、80000条生成与评分及完整输入核验。[扩展报告](artifacts/v73_epsilon_extension_20261111/reports/V73_EPSILON_EXTENSION_REPORT.md)、[新增16组结果](artifacts/v73_epsilon_extension_20261111/reports/extension_results.csv)、[合并50组结果](artifacts/v73_epsilon_extension_20261111/reports/combined_results.csv)、[合并80行配对比较](artifacts/v73_epsilon_extension_20261111/reports/combined_comparisons.csv)。原34组数值复算一致；实际训练预算偏差继续披露。\n'
    status_path.write_text(text)
    print('Completed 16 new configs / 50 combined; reports:',str(report),flush=True)

if __name__=='__main__':main()
