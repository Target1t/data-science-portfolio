"""A prespecified nested-likelihood diagnostic chooses mechanism complexity.

Only the historical no-AI log enters the gate. Evaluation rewards never do.
Chi-square calibration is asymptotic and may fail at finite-sample boundaries.
"""
from pathlib import Path
import json
import numpy as np
from scipy.stats import chi2
from mechanism import design

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'


def main():
    cfg=json.loads((OUT/'config.json').read_text());rows=json.loads((OUT/'records.json').read_text())
    raw=np.load(OUT/'estimates_and_reference.npz')
    threshold=float(chi2.ppf(.95,3));gates=[];extra=[]
    lookup={(r['regime'],r['seed'],r['budget'],r['context'],r['model'],r['rule']):r for r in rows}
    for regime in ['original','shape_stress']:
        for seed in cfg['data_seeds']:
            data=np.load(ROOT/'data'/f'no_ai_{regime}_{seed}.npz')
            for budget in cfg['budgets']:
                for context in range(cfg['contexts']):
                    assert (data['action'][context,:budget]==0).all()
                    w=design(data['x'][context,:budget].reshape(-1,6),0).reshape(-1,5)
                    n=data['origin'][context,:budget].reshape(-1);d=data['down'][context,:budget].reshape(-1)
                    ll=[]
                    for kind in ['logistic','flexible']:
                        q=raw[f'{regime}_{seed}_{budget}_{context}_{kind}__q']
                        p=np.clip(w@q,1e-10,1-1e-10)
                        ll.append(float((d*np.log(p)+(n-d)*np.log1p(-p)).sum()))
                    statistic=max(0.,2*(ll[1]-ll[0]));chosen='flexible' if statistic>threshold else 'logistic'
                    gates.append(dict(regime=regime,seed=seed,budget=budget,context=context,
                                      likelihood_ratio=statistic,threshold=threshold,chosen=chosen))
                    for rule in ['plugin','conservative']:
                        row=lookup[(regime,seed,budget,context,chosen,rule)].copy()
                        row['model']='adaptive';row['source_model']=chosen;extra.append(row)
    merged=rows+extra
    (OUT/'adaptive_gates.json').write_text(json.dumps(gates,indent=2),encoding='utf-8')
    (OUT/'all_records.json').write_text(json.dumps(merged,indent=2),encoding='utf-8')
    summaries=[]
    for regime in ['original','shape_stress']:
        for budget in cfg['budgets']:
            for kind in ['logistic','flexible','adaptive']:
                for rule in ['plugin','conservative']:
                    rr=[r for r in merged if (r['regime'],r['budget'],r['model'],r['rule'])==(regime,budget,kind,rule)]
                    s=dict(regime=regime,budget=budget,model=kind,rule=rule)
                    for m in ['regret','gain','material_harm','near_optimal','q_rmse','value_mae',
                              'simultaneous_gain_coverage','mean_gain_interval_width','no_ai_regret','fit_success','boundary']:
                        s[m]=float(np.mean([r[m] for r in rr]))
                    gg=[g for g in gates if (g['regime'],g['budget'])==(regime,budget)]
                    s['gate_flexible_rate']=float(np.mean([g['chosen']=='flexible' for g in gg]))
                    summaries.append(s)
    (OUT/'all_summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    # Paired game-level differences, retaining all independent log replicates per game.
    comparisons=[]
    for regime in ['original','shape_stress']:
        for budget in cfg['budgets']:
            for baseline in ['logistic','flexible']:
                d=[]
                for context in range(cfg['contexts']):
                    def vals(kind):
                        return [r['regret'] for r in merged if (r['regime'],r['budget'],r['context'],r['model'],r['rule'])==(regime,budget,context,kind,'plugin')]
                    d.append(np.mean(vals('adaptive'))-np.mean(vals(baseline)))
                d=np.array(d);boots=np.random.default_rng(883).choice(d,(4000,len(d))).mean(1)
                comparisons.append(dict(regime=regime,budget=budget,contrast=f'adaptive minus {baseline}',
                    mean=float(d.mean()),ci95=np.quantile(boots,[.025,.975]).tolist()))
    (OUT/'adaptive_comparisons.json').write_text(json.dumps(comparisons,indent=2),encoding='utf-8')
    print('Adaptive selection used only no-AI logs;',len(extra),'evaluation rows added.')


if __name__=='__main__':main()
