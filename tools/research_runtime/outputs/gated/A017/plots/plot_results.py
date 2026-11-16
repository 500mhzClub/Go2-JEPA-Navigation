"""Standalone figures from reviewed A017 results; no new model execution."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from go2nav.research import atomic_json,digest,require_review

b=Path('outputs/gated/A017');out=b/'plots';out.mkdir(exist_ok=False)
for phase in ['calibration','probes']:assert require_review(b/phase/'review.json')['status']=='passed'
c=json.loads((b/'calibration/result.json').read_text());r=json.loads((b/'probes/result.json').read_text())
colors=['#687a90','#147d75'];names=['before','current'];labels=['105,000 steps','220,000 steps']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
fig,axes=plt.subplots(1,3,figsize=(15,4.8),gridspec_kw={'width_ratios':[1,1,1.2]})
modes=['current','future_encoded','rollout','zero','no_rollout'];xt=['Current\nframe','True future\nframe*','4-step\nrollout','Zero future\nactions','No\nrollout']
for ax,ball in zip(axes[:2],['red','white']):
 for i,name in enumerate(names):
  values=[r['cases'][name]['measures'][mode][ball+'_position_rmse_px'] for mode in modes]
  bars=ax.bar(np.arange(len(modes))+(i-.5)*.35,values,.35,color=colors[i],label=labels[i])
  ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=3)
 ax.set(xticks=np.arange(len(modes)),xticklabels=xt,ylabel='Position RMSE (pixels)',title=ball.capitalize()+' ball',ylim=(0,9.5 if ball=='white' else 6.5))
 ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].legend(frameon=False,fontsize=9)
rows=[('current','red','Red: current'),('rollout','red','Red: rollout'),('current','white','White: current'),('rollout','white','White: rollout')]
for i,(mode,ball,label) in enumerate(rows):
 d=r['paired_temporal_gains'][mode][ball];lo,hi=d['ci95']
 axes[2].errorbar(d['mean'],i,xerr=[[d['mean']-lo],[hi-d['mean']]],fmt='o',color=colors[1],capsize=4)
axes[2].set(yticks=range(4),yticklabels=[v[2] for v in rows],xlabel='Earlier minus later error (pixels²)',title='Paired improvement, 95% interval')
axes[2].invert_yaxis();axes[2].axvline(0,color='#777',ls='--',lw=1);axes[2].grid(axis='x',alpha=.15)
fig.suptitle('L1, one environmental ball: 105,000 → 220,000 training steps',fontsize=15)
fig.text(.5,.015,'500 identical validation scenes / 100 episodes; frozen train-fitted readouts. Intervals resample episodes.\n*True future image is encoded only as a scoring reference; it is never supplied to rollout. No planning claim.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.09,1,.93))
for ext in ['png','pdf']:fig.savefig(out/('physical_progress.'+ext),bbox_inches='tight')
plt.close(fig)

rows=[json.loads(line) for line in (b/'snapshots/current/metrics.jsonl').read_text().splitlines()]
train=[x for x in rows if 'validation' not in x];val=[x for x in rows if 'validation' in x]
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
ts=np.array([x['step'] for x in train]);pred=np.array([x['pred'] for x in train]);smooth=np.array([np.median(pred[max(0,i-10):i+1]) for i in range(len(pred))])
axes[0].plot(ts/1000,smooth,label='Train: trailing 11 logs',color=colors[1])
axes[0].plot([x['step']/1000 for x in val],[x['validation']['pred'] for x in val],'o-',label='Validation: saved statistics',color='#c26e36')
axes[0].axvline(105,color='#888',ls='--',lw=1,label='Recovery input')
axes[0].set(yscale='log',xlabel='Training steps (thousands)',ylabel='Logged prediction MSE',title='Original retained learning curve')
axes[0].legend(frameon=False,fontsize=9);axes[0].grid(alpha=.15)
for i,(stage,title) in enumerate([('before','Saved statistics'),('after','Train-calibrated statistics')]):
 values=[c['cases'][name][stage]['val']['saved']['pred_mse'] for name in names]
 bars=axes[1].bar(np.arange(2)+(i-.5)*.32,values,.32,label=title,color=[colors[0],colors[1]][i])
 axes[1].bar_label(bars,labels=[f'{v:.5f}' for v in values],fontsize=9,padding=3)
axes[1].set(yscale='log',xticks=[0,1],xticklabels=labels,ylabel='Prediction MSE on 64 fixed validation windows',title='Frozen-weight BatchNorm diagnostic',ylim=(.002,.23))
axes[1].legend(frameon=False,fontsize=9);axes[1].grid(axis='y',alpha=.15);axes[1].set_axisbelow(True)
fig.suptitle('Calibration changes the apparent prediction gap',fontsize=15)
fig.text(.5,.015,'Calibration uses 4,096 training windows only; learned weights remain unchanged.\nLatent MSE is not a physical-error or planning metric. The two panels use different diagnostic samples.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.09,1,.93))
for ext in ['png','pdf']:fig.savefig(out/('calibration_and_curves.'+ext),bbox_inches='tight')
plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
for j,name in enumerate(names):
 arr=np.load(b/'probes'/(name+'_cases.npz'))
 for i,ball in enumerate(['red','white']):
  ax=axes[i,j];ax.scatter(arr[ball+'_distance'],arr['latent_distance'],s=5,alpha=.18,color=colors[j],rasterized=True)
  rho=r['cases'][name]['geometry'][ball+'_spearman'];ax.set_title(f'{labels[j]} · {ball} · Spearman ρ={rho:.3f}')
  if i==1:ax.set_xlabel('Physical position distance (pixels)')
  if j==0:ax.set_ylabel('Latent Euclidean distance')
fig.suptitle('Weaker Euclidean geometry can coexist with better position readout',fontsize=14)
fig.text(.5,.015,'2,000 identical cross-episode pairs from 500 scenes. Pairs share endpoints; descriptive correlations only.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.05,1,.94))
for ext in ['png','pdf']:fig.savefig(out/('latent_geometry.'+ext),bbox_inches='tight')
plt.close(fig)
atomic_json(out/'manifest.json',dict(research_id='A017',calibration_review_sha256=digest(b/'calibration/review.json'),
    probes_review_sha256=digest(b/'probes/review.json'),script_sha256=digest(__file__),
    artifacts={p.name:digest(p) for p in out.iterdir() if p.suffix in ['.png','.pdf']}))
print('Created three reviewed-result figures as PNG and PDF.')
