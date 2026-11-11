#!/usr/bin/env python3
"""Persistent, resumable training -> generation -> scoring -> report pipeline."""
import fcntl
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import run_train_grid as train
import run_eval_grid as evaluate

ROOT=Path(__file__).resolve().parent
PY=train.PY
STATE={}
LOCK=threading.RLock()
TRAIN_GPUS=[5,7]
EVAL_GPU=4

def utc():return datetime.now(timezone.utc).isoformat()
def write(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
def update(name,**fields):
    with LOCK:
        STATE.setdefault(name,{'name':name}).update(fields)
        STATE[name]['updated_utc']=utc()
        write(ROOT/'pipeline_status.json',{'updated_utc':utc(),'pid':os.getpid(),'configs':list(STATE.values())})
        print(json.dumps(STATE[name],ensure_ascii=False),flush=True)
def available(gpus,required):
    result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    free={int(x.split(',')[0]):int(x.split(',')[1]) for x in result.stdout.splitlines()}
    return all(free[g]>=required for g in gpus)
def wait_for_memory(gpus,required,name,stage):
    while not available(gpus,required):
        update(name,**{stage:'waiting_for_gpu_memory'},waiting_gpus=gpus,required_free_mib=required)
        time.sleep(60)

def eval_worker(jobs):
    while True:
        cfg=jobs.get()
        if cfg is None:jobs.task_done();return
        name=cfg['name']
        try:
            for attempt in range(1,4):
                wait_for_memory([EVAL_GPU],36000,name,'evaluation')
                update(name,evaluation='running',eval_attempt=attempt,eval_gpu=EVAL_GPU)
                result=evaluate.eval_one(cfg,str(EVAL_GPU))
                (ROOT/'eval'/name).mkdir(parents=True,exist_ok=True)
                write(ROOT/'eval'/name/'status.json',{**result,'attempt':attempt,'updated_utc':utc()})
                if result['complete']:
                    update(name,evaluation='completed',scored_samples=5000);break
                update(name,evaluation='failed',eval_failure=result)
                if attempt<3:time.sleep(60)
        except Exception:
            update(name,evaluation='failed',eval_error=traceback.format_exc())
        finally:jobs.task_done()

def main():
    with (ROOT/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert json.loads((ROOT/'audit/frozen_reuse_checks.json').read_text())['status']=='passed'
        assert json.loads((ROOT/'audit/trainer_loss_checks.json').read_text())['status']=='passed'
        configs=json.loads((ROOT/'v7_grid_manifest.json').read_text())['configs'];assert len(configs)==16
        for c in configs:STATE[c['name']]={'name':c['name'],'training':'queued','evaluation':'queued'}
        write(ROOT/'pipeline_status.json',{'pid':os.getpid(),'started_utc':utc(),'configs':list(STATE.values())})
        metadata=json.loads((ROOT/'run_metadata.json').read_text())
        metadata.update({'status':'running','started_utc':metadata.get('started_utc',utc()),'pipeline_pid':os.getpid(),'gpu_assignment':{'training':TRAIN_GPUS,'evaluation':EVAL_GPU}})
        write(ROOT/'run_metadata.json',metadata)
        q=queue.Queue();worker=threading.Thread(target=eval_worker,args=(q,),daemon=True);worker.start()
        for cfg in configs:
            name=cfg['name']
            for attempt in range(1,4):
                if not train.finished(ROOT/'outputs'/name):wait_for_memory(TRAIN_GPUS,28000,name,'training')
                update(name,training='running',train_attempt=attempt,train_gpus=TRAIN_GPUS)
                log=ROOT/'logs'/f'train_{name}.log'
                status=ROOT/'outputs'/name/'status.json'
                if log.exists() and not train.finished(ROOT/'outputs'/name):
                    stamp=str(time.time_ns());shutil.copy2(log,log.with_name(log.stem+f'_before_{stamp}.log'))
                    if status.exists():shutil.copy2(status,status.with_name(f'status_before_{stamp}.json'))
                result=train.train_one(cfg,','.join(map(str,TRAIN_GPUS)),30071)
                if result['status'] in ['completed','skipped_existing']:
                    update(name,training='completed',checkpoint=str(ROOT/'outputs'/name/'checkpoints_dpo/final_hf'))
                    q.put(cfg);break
                update(name,training='failed',train_failure=result)
                if attempt<3:time.sleep(60)
        q.put(None);q.join();worker.join()
        write(ROOT/'train_status.json',[{'name':s['name'],'status':s['training']} for s in STATE.values()])
        write(ROOT/'eval_status.json',[{'name':s['name'],'complete':s['evaluation']=='completed','samples':s.get('scored_samples',0)} for s in STATE.values()])
        all_done=all(s['training']=='completed' and s['evaluation']=='completed' for s in STATE.values())
        if not all_done:raise RuntimeError('Some configs failed after retries; inspect pipeline_status.json')
        update('_analysis',analysis='running')
        with (ROOT/'logs/finalize.log').open('a') as f:
            result=subprocess.run([str(PY),str(ROOT/'finalize_extension.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Final validation/report failed; inspect logs/finalize.log')
        update('_analysis',analysis='completed')

if __name__=='__main__':
    try:main()
    except Exception:
        print(traceback.format_exc(),flush=True)
        write(ROOT/'pipeline_failure.json',{'utc':utc(),'traceback':traceback.format_exc()})
        sys.exit(1)
