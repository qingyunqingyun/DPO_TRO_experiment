#!/usr/bin/env python3
"""Validate all dev scores, export complete tables and a reproducible report."""
import csv, hashlib, json, math, re, statistics, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parent
def read(p):return [json.loads(s) for s in p.open() if s.strip()]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def csvout(p,rows):
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def hashfile(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    m=json.loads((ROOT/'v7_grid_manifest.json').read_text()); configs=m['configs']
    dev=read(ROOT/'data/pairs/dev.jsonl'); expected={r['prompt_id']:r for r in dev}
    audit=json.loads((ROOT/'audit/input_and_math_checks.json').read_text())
    keys=sorted(expected); vectors={}; counts={}; score_hashes={}; prompt_rows=[]
    tok=AutoTokenizer.from_pretrained(m['rm'],local_files_only=True)
    truncation=[]
    for c in configs:
        name=c['name']; p=ROOT/f'eval/{name}/scored.jsonl'; rows=read(p)
        assert len(rows)==5000,(name,len(rows))
        groups=defaultdict(list); seen=set(); maxlen=0
        for r in rows:
            assert r['prompt_id'] in expected and r['pair_id']==expected[r['prompt_id']]['pair_id']
            assert r['prompt']==expected[r['prompt_id']]['prompt']
            assert r['seed']==2101 and 0<=r['sample_id']<10
            k=(r['prompt_id'],r['sample_id']); assert k not in seen;seen.add(k)
            v=float(r['skywork_score']);assert math.isfinite(v)
            groups[r['prompt_id']].append(v)
        # Tokenize the full, untruncated RM input with the exact scorer rendering.
        texts=[]
        for r in rows:
            t=tok.apply_chat_template(r['prompt']+r['response'],tokenize=False,add_generation_prompt=False)
            if tok.bos_token and t.startswith(tok.bos_token):t=t[len(tok.bos_token):]
            texts.append(t)
        actual=[]
        for i in range(0,len(texts),128):
            encoded=tok(texts[i:i+128],padding=False,truncation=False)
            actual.extend(map(len,encoded['input_ids']))
        assert max(actual)<=4096,(name,max(actual))
        assert all(int(r['token_length'])==a for r,a in zip(rows,actual)),name
        truncation.append({'config':name,'samples':len(rows),'full_rm_max_tokens':max(actual),'rm_truncated_samples':0})
        assert set(groups)==set(keys) and all(len(v)==10 for v in groups.values())
        vectors[name]=np.array([np.mean(groups[k]) for k in keys]); counts[name]=len(rows)
        score_hashes[name]={'path':str(p),'sha256':hashfile(p),'bytes':p.stat().st_size}
        print('validated',name,flush=True)
    oracle=vectors['oracle_soft_dpo']; shortages={n:np.maximum(oracle-v,0) for n,v in vectors.items()}
    rng=np.random.default_rng(601); idx=rng.integers(0,len(keys),size=(2000,len(keys)))
    def ci(v):
        vals=v[idx].mean(axis=1);return [float(x) for x in np.quantile(vals,[.025,.975])]
    result=[]
    for c in configs:
        n=c['name'];v=vectors[n];s=shortages[n];sci=ci(s);rci=ci(v)
        result.append({'config':n,'method':c['method'],'distance':c['distance'] or '', 'rho':c['rho'],'epsilon':c['epsilon'],
            'prompts':500,'samples':5000,'mean_reward':float(v.mean()),'reward_ci95_low':rci[0],'reward_ci95_high':rci[1],
            'oracle_shortfall':float(s.mean()),'shortfall_ci95_low':sci[0],'shortfall_ci95_high':sci[1],
            'gap_to_oracle':float((v-oracle).mean())})
        for k,r,short in zip(keys,v,s):prompt_rows.append({'config':n,'prompt_id':k,'mean_reward':float(r),'oracle_mean_reward':float(oracle[keys.index(k)]),'positive_shortfall':float(short)})
    comparisons=[]
    for c in configs:
        if c['method']!='tro_trust_common':continue
        n=c['name'];dro=next(x['name'] for x in configs if x['method']=='label_dro_common' and x['distance']==c['distance'] and x['rho']==c['rho'])
        for base in [dro,'soft_dpo']:
            dr=vectors[n]-vectors[base];ds=shortages[n]-shortages[base];rci=ci(dr);sci=ci(ds)
            comparisons.append({'config':n,'distance':c['distance'],'rho':c['rho'],'epsilon':c['epsilon'],'baseline':base,
                'delta_reward':float(dr.mean()),'delta_reward_ci95_low':rci[0],'delta_reward_ci95_high':rci[1],
                'delta_shortfall':float(ds.mean()),'delta_shortfall_ci95_low':sci[0],'delta_shortfall_ci95_high':sci[1]})
    dest=ROOT/'reports';dest.mkdir(exist_ok=True)
    csvout(dest/'dev_grid_results.csv',result);csvout(dest/'tro_baseline_comparisons.csv',comparisons);csvout(dest/'prompt_scores.csv',prompt_rows)
    csvout(dest/'length_audit.csv',truncation)
    save(ROOT/'eval/v7_dev_metrics.json',{'oracle_samples':5000,'oracle_mean_reward':float(oracle.mean()),'bootstrap_replicates':2000,'bootstrap_seed':601,'confidence_interval_unit':'prompt','results':result,'comparisons':comparisons})
    save(ROOT/'audit/eval_checks.json',{'status':'passed','configs':34,'samples_per_config':5000,'total_samples':170000,'prompt_count':500,'all_sample_ids_unique':True,'matching_prompt_text':True,'all_rewards_finite':True,'rm_truncated_samples':0,'max_rm_tokens':max(x['full_rm_max_tokens'] for x in truncation),'score_files':score_hashes})
    # Preserve the original attempt status and reconcile successful exports.
    trainstat=ROOT/'train_status.json'
    if not (ROOT/'audit/train_initial_attempt_status.json').exists():
        (ROOT/'audit/train_initial_attempt_status.json').write_bytes(trainstat.read_bytes())
    final_status=[]
    for c in configs:
        p=ROOT/'outputs'/c['name'];hf=p/'checkpoints_dpo/final_hf'
        assert (hf/'config.json').exists() and any(hf.glob('*.safetensors'))
        retried=ROOT/f"logs/retry_{c['name']}.log"
        log=retried if retried.exists() else ROOT/f"logs/train_{c['name']}.log"
        t=log.read_text(errors='replace')
        assert 'exits successfully' in t and re.search(r'Train epoch: 100%.*?2/2',t),(c['name'],str(log))
        original=json.loads((p/'status.json').read_text())
        rec={'name':c['name'],'status':'completed','checkpoint':str(hf),'successful_log':str(log),'retried':retried.exists(),'initial_returncode':original.get('returncode')}
        final_status.append(rec)
        initial=p/'status_initial_attempt.json'
        if not initial.exists():initial.write_text(json.dumps(original,indent=2)+'\n')
        save(p/'status.json',{**original,**rec,'returncode':0,'note':'Reconciled using successful 2-epoch log and validated safetensors export.'})
    save(trainstat,final_status)
    save(ROOT/'eval_status.json',[{'name':c['name'],'complete':True,'samples':5000,'scored':str(ROOT/f"eval/{c['name']}/scored.jsonl")} for c in configs])
    initial_status=json.loads((ROOT/'audit/train_initial_attempt_status.json').read_text())
    started=min(x['start_time'] for x in initial_status)
    trained=max(x['modified_utc_timestamp'] for x in json.loads((ROOT/'audit/model_exports.json').read_text()))
    scored_at=max((ROOT/f"eval/{c['name']}/scored.jsonl").stat().st_mtime for c in configs)
    def utc(ts):return datetime.fromtimestamp(ts,timezone.utc).isoformat()
    cost={'first_full_train_start_utc':utc(started),'last_model_export_utc':utc(trained),'last_scoring_write_utc':utc(scored_at),'train_wall_hours':(trained-started)/3600,'evaluation_and_interruption_wall_hours':(scored_at-trained)/3600,'train_to_scoring_wall_hours':(scored_at-started)/3600,'note':'Wall times include initialization, contention, failures and retries. GPU active compute-hours were not instrumented; no claim equating occupied time to utilization.'}
    save(ROOT/'audit/timing.json',cost)
    metadata=json.loads((ROOT/'run_metadata.json').read_text())
    metadata.update({'status':'completed','finished_scoring_utc':utc(scored_at),'timing':cost,'oracle_temperature':audit['oracle_temperature'],'training':{'epochs':2,'dp_world_size':2,'microbatch_pairs_per_rank':1,'gradient_accumulation':64,'effective_batch_pairs':128,'optimizer_updates':31,'scheduler_planned_updates':30,'final_uncommitted_pairs':32},'hashes':{'input':'audit/input_and_math_checks.json','initial_models_and_tokenizers':'audit/initial_model_hashes.json','trained_models':'audit/model_exports.json','scores':'audit/eval_checks.json'},'checks':{'inputs_and_math':'audit/input_and_math_checks.json','trainer':'audit/trainer_loss_checks.json','eval':'audit/eval_checks.json'},'complete_configs':34,'scored_generations':170000})
    save(ROOT/'run_metadata.json',metadata)
    obs=read(ROOT/'data/observations/base_train.jsonl'); diag=[]
    for n in [1,3,5]:
        for k in range(n+1):
            ct=sum(r['n_votes']==n and r['k_1']==k for r in obs)
            diag.append({'n_votes':n,'k_1':k,'pairs':ct})
    csvout(dest/'vote_patterns.csv',diag)
    shortbest=min(result,key=lambda x:x['oracle_shortfall'] if x['config']!='oracle_soft_dpo' else float('inf'))
    rewardbest=max([x for x in result if x['config']!='oracle_soft_dpo'],key=lambda x:x['mean_reward'])
    now=datetime.now(timezone.utc).isoformat()
    lines=['# v7.2 KL / W1 全网格开发集实验报告','',f'协议 commit：`7beb6b4`；生成时间：{now}。',
       '', '**运行范围：34/34 个配置完成训练、生成和评分。每个配置 500 道相同开发题，每题 10 个生成，共 170,000 条评分。**',
       '',f"完整训练开始 {cost['first_full_train_start_utc']}，最后模型导出 {cost['last_model_export_utc']}，最后评分写入 {cost['last_scoring_write_utc']}；训练到评分合计 {cost['train_to_scoring_wall_hours']:.2f} 小时（含初始化、等待、失败与补跑），训练阶段 {cost['train_wall_hours']:.2f} 小时，随后评测阶段 {cost['evaluation_and_interruption_wall_hours']:.2f} 小时。未测量 GPU 实际算力利用小时。",
       '',f"除 oracle 参照外，平均奖励最高为 `{rewardbest['config']}`：{rewardbest['mean_reward']:.6f}；oracle 短缺最小为 `{shortbest['config']}`：{shortbest['oracle_shortfall']:.6f}。这些是开发集描述性排序。",'',
       '## 参数与数据', '',
       '| 参数 | 实际值 |','|---|---|',
       '| 模型 / 参考模型 | Llama-3.2-1B-Instruct，同初始化、冻结参考 |',
       '| oracle / 评测 RM | Skywork/Skywork-Reward-V2-Llama-3.1-8B，BF16 |',
       '| 固定数据 | 2000 train pairs；500 dev prompts；无 prompt 重叠 |',
       '| N_i 与配额 | 1/3/5：800/600/600 对（40%/30%/30%）；共 5600 票 |',
       '| allocation / vote / train / generation seeds | 1731 / 2101 / 1101 / 2101 |',
       f'| 偏好温度 | T_pref={audit["oracle_temperature"]}，训练集非零绝对 RM 分差中位数 |',
       '| 后验先验 / epsilon | Beta(0.5,0.5) / {0.025,0.05,0.1} |',
       '| KL 半径 | {0.005,0.05,0.1,0.3}；KL(Ber(h) || Ber(p)) |',
       '| W1 半径 | {0.01,0.05,0.15,0.3}；d(0,1)=1 |',
       '| 优化 | 2 epochs；lr=5e-7；beta=0.01；AdamW betas=(0.9,0.95)，wd=0，warmup=0.1，clip=1 |',
       '| 并行 / batch | ZeRO-2，DP=2，每卡 microbatch=1，累积=64，有效 batch=128 |',
       '| 数值 / 长度 | BF16 模型，FP32 logprob/loss；训练每侧 max_len=2048，完整回答对 |',
       '| 生成 | G=10，temperature=0.7，top_p=0.9，max_new_tokens=1024，context=4096，vLLM memory utilization=0.4 |',
       '| 评分 | responses 模式，batch=2，max_length=4096，实际完整文本无截断 |',
       '| 预算 | 每配置 5000 条；全网格 170000 条；单次投票/训练/生成 seed |',
       '', '## 完整结果', '',
       'R 为 500 道题各自 10 次奖励均值的平均；S_O=mean_i max(R_oracle,i-R_method,i,0)，越小越好。Oracle policy 是精确概率标签训练的参照，不是理论上界。',
       '', '| 配置 | 距离 | rho | epsilon | R ↑ | S_O ↓ | S_O 95% CI |', '|---|---|---:|---:|---:|---:|---|']
    for r in result:
        lines.append(f"| {r['config']} | {r['distance'] or '—'} | {r['rho'] if r['rho'] is not None else '—'} | {r['epsilon'] if r['epsilon'] is not None else '—'} | {r['mean_reward']:.6f} | {r['oracle_shortfall']:.6f} | [{r['shortfall_ci95_low']:.6f}, {r['shortfall_ci95_high']:.6f}] |")
    lines+=['','## TRO 相对基线的配对比较','','ΔR=TRO−baseline 越大越好；ΔS_O=TRO−baseline 越小越好。95% CI 按共同 prompt 配对 bootstrap 2000 次，seed=601；未作多重比较校正。','',
       '| TRO 配置 | baseline | ΔR | ΔR 95% CI | ΔS_O | ΔS_O 95% CI |','|---|---|---:|---|---:|---|']
    for r in comparisons:
        lines.append(f"| {r['config']} | {r['baseline']} | {r['delta_reward']:+.6f} | [{r['delta_reward_ci95_low']:+.6f}, {r['delta_reward_ci95_high']:+.6f}] | {r['delta_shortfall']:+.6f} | [{r['delta_shortfall_ci95_low']:+.6f}, {r['delta_shortfall_ci95_high']:+.6f}] |")
    lines+=['','## 观察到的方向','','| 距离 | 对比 | 配置数 | ΔR>0 | ΔS_O<0 | 两者同时成立 |','|---|---|---:|---:|---:|---:|']
    for d in ['kl','w1']:
        for b in ['same_radius_dro','soft_dpo']:
            cc=[x for x in comparisons if x['distance']==d and (x['baseline']=='soft_dpo')==(b=='soft_dpo')]
            lines.append(f"| {d} | {b} | {len(cc)} | {sum(x['delta_reward']>0 for x in cc)} | {sum(x['delta_shortfall']<0 for x in cc)} | {sum(x['delta_reward']>0 and x['delta_shortfall']<0 for x in cc)} |")
    lines+=['','## 验证、运行异常与解释范围','',
       f'- 输入审计通过：训练与 dev 无重复 prompt；实际训练每侧最长 {audit["max_training_side_tokens"]} tokens；dev 最长模板 prompt 为 {audit["max_dev_prompt_tokens_including_generation_template"]} tokens，生成预算在 4096 context 内。',
       f'- 所有 170000 条生成逐条核验：sample_id 0…9 唯一且齐全；RM 实际完整输入最长 {max(x["full_rm_max_tokens"] for x in truncation)} tokens，无 RM 截断，全部奖励有限。',
       '- 数学审计通过：KL/W1 端点、lambda 与协议参考表一致、标签交换对称、跨半径权重不变，以及 loss/梯度和退化关系。普通训练数据 p_star 为占位值 0.5，实际 p_star 只用于投票模拟与 oracle trainer。',
       '- 继承原 trainer 的尾批策略：每 rank 每 epoch 1000 个 microstep，累积跨 epoch 延续；4000 次回答对曝光对应 31 个完整 optimizer update（3968 对），末尾 32 对的累积未提交；scheduler 计划 30 update，实际 31。所有配置一致。这是本轮相对严格 4000 对更新预算的偏差，未在结果出来后修改。',
       '- 首轮外部 vLLM 显存占用，以及原线程队列可能把不同配置重叠分配到同一组 GPU，导致初始化 OOM；7 个配置以相同数据和训练 seed 从初始化重新训练成功，补跑使用 GPU 1,3。调度代码已修正为固定每组 GPU 串行队列。另有成功训练被错误路径检查标为 failed，已保留原始状态并按成功日志/模型导出纠正。',
       '- 生成/评分脚本最初有路径错误，生成默认 KV cache 也发生 OOM；已纠正路径并使用 vLLM utilization=0.4 完成，两个基线保留完整生成并单独补评分。原始失败及补跑日志保留。',
       '- dev 额外生成了投票字段用于 trainer 内部的 dev loss，正式生成指标完全不依赖这些字段。该实现与协议“开发生成不分配人数”文字存在差异，但未改变 500 道题、生成或 RM 指标。',
       '- 原生成脚本未保存 vLLM finish_reason，无法精确回报达到 max_new_tokens 的比例；不将长度估计冒充精确统计。',
       '- 置信区间反映本次已生成样本在 prompt 间的变异；不覆盖重新抽票、训练随机性及多重选参。当前只有一次 seed 重复，500 题属于开发集，不能当作独立测试证明。没有读取或启动 final test；人数增长及先验敏感性训练不属于本轮 34 配置网格。',
       '', '## 交付文件','',
       '- `dev_grid_results.csv`：34 行完整参数与结果。',
       '- `tro_baseline_comparisons.csv`：48 行相对同半径 DRO / soft DPO 的配对差值及区间。',
       '- `prompt_scores.csv`：17000 行配置×prompt 平均奖励及短缺。',
       '- `length_audit.csv`、`vote_patterns.csv`：逐配置完整输入长度与票型数量。',
       '- `../eval/<config>/scored.jsonl`：逐生成完整回答及 RM 分数；`../outputs/<config>/checkpoints_dpo/final_hf`：模型。',
       '- `../v7_grid_manifest.json`、`../run_metadata.json`、`../audit/`：冻结配置、种子、源代码快照与校验。',
       '- `../train_status.json`、`../eval_status.json`：最终状态；原始训练状态保留在 `../audit/train_initial_attempt_status.json`。','']
    (dest/'V7_DEV_GRID_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'report':str(dest/'V7_DEV_GRID_REPORT.md'),'best_reward':rewardbest,'best_shortfall':shortbest},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
