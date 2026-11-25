"""Create publication-style figures and a self-contained HTML research brief."""
from pathlib import Path
import html,json,base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import markdown
from simulator import deterministic
from models import Dynamics,rollout

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';FIG=OUT/'figures'
NAMES={'no_feedback':'No population feedback','direct':'Direct population MLP',
       'dense_kernel':'Dense transition MLP','structured':'Structured one-step',
       'structured_rollout':'Structured multi-step'}
COLORS=dict(zip(NAMES,['#87929c','#e6a03c','#b373a8','#459ea6','#163e69']))
SPLITS={'iid_static':'New static scenarios','initial_shift':'Initial-population shift','ai_switch':'Mid-trajectory AI switch'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':140,'savefig.dpi':200})


def dumpfig(fig,name):
    fig.savefig(FIG/f'{name}.png',bbox_inches='tight')
    fig.savefig(FIG/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)


def main():
    cfg=json.loads((OUT/'config.json').read_text());rows=json.loads((OUT/'metrics.json').read_text())
    manifest=json.loads((OUT/'manifest.json').read_text());seeds=cfg['seeds']
    comparisons=json.loads((OUT/'paired_comparisons.json').read_text())
    def get(kind,split,metric):return np.array([r[metric] for r in rows if r['model']==kind and r['split']==split])
    def pred(kind,seed):return np.load(OUT/'predictions'/f'{kind}_{seed}.npz')

    # All scenarios; uncertainty band displays training-seed variation, not a confidence interval.
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for ax,(split,title) in zip(axes,SPLITS.items()):
        for kind in NAMES:
            curves=np.stack([pred(kind,s)[f'{split}__tv'].mean(0) for s in seeds])
            mean=curves.mean(0);sd=curves.std(0)
            ax.plot(mean,color=COLORS[kind],label=NAMES[kind],lw=2)
            ax.fill_between(np.arange(len(mean)),np.maximum(0,mean-sd),mean+sd,color=COLORS[kind],alpha=.12)
        ax.set_title(title);ax.set_xlabel('Forecast horizon (periods)');ax.grid(alpha=.15)
        if split=='ai_switch':ax.axvline(cfg['switch_time'],color='black',ls=':',lw=1)
    axes[0].set_ylabel('Population total variation error')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.12),frameon=False)
    fig.suptitle('Forecast errors on independent finite-population trajectories')
    fig.tight_layout();dumpfig(fig,'01_forecast_errors')

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,split in zip(axes,['iid_static','initial_shift']):
        for i,kind in enumerate(NAMES):
            vals=get(kind,split,'mean_regret')
            ax.bar(i,vals.mean(),color=COLORS[kind],width=.65)
            ax.scatter(np.full(len(vals),i)+np.linspace(-.1,.1,len(vals)),vals,color='black',s=15,zorder=3)
        ax.set_xticks(range(5),['No feedback','Direct','Dense','Structured','Multi-step'],rotation=25)
        ax.set_title(SPLITS[split]);ax.set_ylabel('Mean decision regret (level units)');ax.grid(axis='y',alpha=.2)
    fig.suptitle(f'AI placement: independent {cfg["mc_replicates"]}-replicate Monte Carlo reference\nDots = three training seeds; lower is better')
    fig.tight_layout();dumpfig(fig,'02_decision_regret')

    # An a-priori case, not selected by model performance: paper Figure 4 parameters.
    params=np.repeat([[1.,1.3]],6,axis=0);x0=np.ones((6,6))/6
    acts=np.arange(6)[:,None].repeat(60,axis=1)
    ref=deterministic(params,x0,acts)
    model=Dynamics('structured_rollout')
    model.load_state_dict(torch.load(OUT/'models'/f'structured_rollout_{seeds[0]}.pt',weights_only=True)['state_dict'])
    model.eval()
    with torch.no_grad(): learned=rollout(model,torch.tensor(x0,dtype=torch.float32),
        torch.tensor(params,dtype=torch.float32),torch.tensor(acts,dtype=torch.float32)).numpy()
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for ax,indices,label in zip(axes,[[0],[2,3],[4,5]],['Inactive: level 0','Middle: levels 2–3','High: levels 4–5']):
        for action,color in [(0,'#87929c'),(3,'#163e69'),(4,'#d57c38')]:
            ax.plot(ref[action][:,indices].sum(1),color=color,label=f'AI={action}, formula')
            ax.plot(learned[action][:,indices].sum(1),color=color,ls='--',label=f'AI={action}, learned')
        ax.set_title(label);ax.set_xlabel('Period');ax.set_ylim(0,1);ax.grid(alpha=.15)
    axes[0].set_ylabel('Human population share')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc='lower center',bbox_to_anchor=(.5,-.14),frameon=False)
    fig.suptitle('CSWIM-linked case: alpha = 1, dc = 1.3; uniform initial population\nFormula = deterministic mean-field rollout, not the old penalized steady state')
    fig.tight_layout();dumpfig(fig,'03_cswim_population_mechanism')

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for kind in NAMES:
        axes[0].scatter(get(kind,'initial_shift','one_step_tv'),get(kind,'initial_shift','rollout_tv'),
                        color=COLORS[kind],label=NAMES[kind],s=50)
        axes[1].scatter(get(kind,'initial_shift','rollout_tv'),get(kind,'initial_shift','mean_regret'),
                        color=COLORS[kind],s=50)
    axes[0].set_xlabel('One-step TV error');axes[0].set_ylabel('60-period mean TV error')
    axes[1].set_xlabel('60-period mean TV error');axes[1].set_ylabel('Mean decision regret')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Prediction quality and decision quality are distinct: initial-population shift')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='lower center',bbox_to_anchor=(.5,-.12),frameon=False)
    fig.tight_layout();dumpfig(fig,'04_prediction_vs_decision')

    summary=[]
    for split in SPLITS:
        for kind in NAMES:
            entry=dict(split=split,model=kind)
            for metric in ['one_step_tv','rollout_tv','final_tv','mean_regret','within_002_rate']:
                if any(metric in r for r in rows if r['split']==split):
                    vals=get(kind,split,metric)
                    entry[metric+'_mean']=float(vals.mean());entry[metric+'_sd']=float(vals.std(ddof=1))
            summary.append(entry)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    csv_text='split,model,one_step_tv,rollout_tv,mean_regret,within_002_rate\n'
    for e in summary:
        csv_text+=','.join(str(e.get(k,'')) for k in ['split','model','one_step_tv_mean','rollout_tv_mean','mean_regret_mean','within_002_rate_mean'])+'\n'
    (OUT/'summary.csv').write_text(csv_text,encoding='utf-8')

    def table(split):
        txt='<table><tr><th>Model</th><th>One-step TV</th><th>Rollout TV</th><th>Regret</th><th>Within 0.02</th></tr>'
        for e in summary:
            if e['split']!=split:continue
            def fmt(k):
                return f"{e[k+'_mean']:.4f} ± {e[k+'_sd']:.4f}" if k+'_mean' in e else '—'
            txt+=f"<tr><td>{NAMES[e['model']]}</td><td>{fmt('one_step_tv')}</td><td>{fmt('rollout_tv')}</td><td>{fmt('mean_regret')}</td><td>{fmt('within_002_rate')}</td></tr>"
        return txt+'</table>'
    imgs=''.join(f'<figure><img src="data:image/png;base64,{base64.b64encode((FIG/(name+".png")).read_bytes()).decode()}"><figcaption>{caption}</figcaption></figure>'
                 for name,caption in [('01_forecast_errors','Shading shows ±1 training-seed SD; the dotted line marks the AI switch.'),
                  ('02_decision_regret','Monte Carlo best-of-six reference; selection noise is quantified in manifest.json.'),
                  ('03_cswim_population_mechanism','Preselected paper parameter case. No stationary-equilibrium claim.'),
                  ('04_prediction_vs_decision','Each point is one training seed. Associations do not establish causality.')])
    # The written narrative lives in REPORT.md and is embedded verbatim, so it cannot drift.
    narrative=(ROOT/'REPORT.md').read_text(encoding='utf-8') if (ROOT/'REPORT.md').exists() else 'Detailed analysis will be saved in REPORT.md.'
    html_text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CSWIM Extension | Feedback-Aware Player Dynamics</title><style>
body{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;color:#193246;background:#eef3f7;margin:0}main{max-width:1120px;margin:auto;background:white;padding:42px}h1{font-size:32px;line-height:1.3}h2{margin-top:36px}p{max-width:950px}.tag{color:#476b86;letter-spacing:1px;font-size:13px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px 8px;border-bottom:1px solid #dbe4ec;text-align:left}th{background:#eaf1f6}figure{margin:32px 0}img{width:100%;height:auto}figcaption{font-size:13px;color:#566b7c}pre{white-space:pre-wrap;font:14px/1.7 system-ui,"Microsoft YaHei",sans-serif;background:#f7f9fb;padding:22px}a{color:#176497}@media(max-width:700px){main{padding:18px}table{font-size:11px}th,td{padding:5px}}
</style><main><p class="tag">CSWIM EXTENSION · REPRODUCIBLE COMPUTATIONAL STUDY</p>
<h1>Feedback-Aware Learning of<br>AI–Human Player Dynamics</h1>
<p>Learning population-dependent transitions from synthetic player trajectories while preserving the original CSWIM Markov mechanism.</p>
<p><b>Scope:</b> synthetic data, known context parameters, fixed six-level state space, finite-horizon evaluation. This is a research prototype, not real-player validation or a publication claim.</p>'''
    html_text+=f'<p>Training: {cfg["train_scenarios"]} parameter scenarios × 6 AI placements × {cfg["train_horizon"]} periods. Population: {cfg["population"]}. Training seeds: {seeds}. Test: {cfg["test_scenarios"]} new parameter scenarios per condition, {cfg["test_horizon"]} periods.</p>'
    for split,title in SPLITS.items():html_text+=f'<h2>{title}</h2>'+table(split)
    html_text+='<p>Entries are mean ± sample SD across three training seeds, not confidence intervals. Regret is measured in mean human-level units.</p>'+imgs
    html_text+='<h2>Research report</h2>'+markdown.markdown(narrative,extensions=['tables','fenced_code'])+'</main></html>'
    (ROOT/'research_brief.html').write_text(html_text,encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print('Figure and HTML report generation complete.')


if __name__=='__main__':main()
