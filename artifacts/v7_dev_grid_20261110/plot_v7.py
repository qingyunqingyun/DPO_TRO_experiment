#!/usr/bin/env python3
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent
rows=list(csv.DictReader((ROOT/'reports/dev_grid_results.csv').open()))
fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
for col,d in enumerate(['kl','w1']):
    rs=[r for r in rows if r['distance']==d];rhos=sorted(set(float(r['rho']) for r in rs))
    for row,(metric,label,cmap) in enumerate([('mean_reward','Mean RM reward (higher is better)','viridis'),('oracle_shortfall','Oracle shortfall (lower is better)','viridis_r')]):
        allvals=[float(r[metric]) for r in rows if r['distance']]
        values=np.zeros((4,4))
        for i,rho in enumerate(rhos):
            for j,eps in enumerate([None,.025,.05,.1]):
                r=next(r for r in rs if float(r['rho'])==rho and (float(r['epsilon']) if r['epsilon'] else None)==eps)
                values[i,j]=float(r[metric])
        ax=axs[row,col];im=ax.imshow(values,aspect='auto',vmin=min(allvals),vmax=max(allvals),cmap=cmap)
        ax.set_xticks(range(4),['DRO','TRO eps=.025','TRO eps=.05','TRO eps=.1'])
        ax.set_yticks(range(4),[f'{r:g}' for r in rhos]);ax.set_ylabel('Common radius')
        ax.set_title(d.upper()+': '+label)
        for i in range(4):
            for j in range(4):
                x=(values[i,j]-min(allvals))/max(1e-12,max(allvals)-min(allvals))
                if row==1:x=1-x
                ax.text(j,i,f'{values[i,j]:.4f}',ha='center',va='center',color='white' if x<.5 else 'black',fontsize=10)
        fig.colorbar(im,ax=ax,shrink=.8)
soft=next(r for r in rows if r['config']=='soft_dpo');oracle=next(r for r in rows if r['config']=='oracle_soft_dpo')
fig.suptitle(f"v7.2 dev grid: 500 prompts, 10 samples per prompt, one training seed\nSoft DPO: R={float(soft['mean_reward']):.4f}, S={float(soft['oracle_shortfall']):.4f}; oracle: R={float(oracle['mean_reward']):.4f}",fontsize=13)
fig.savefig(ROOT/'reports/dev_grid_heatmap.png',dpi=180)
fig.savefig(ROOT/'reports/dev_grid_heatmap.pdf')
plt.close(fig)
