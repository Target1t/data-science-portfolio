"""Stage 2: no-AI logs -> latent response recovery -> unseen-AI deployment."""
from pathlib import Path
import sys,json,time,csv,hashlib,argparse
import numpy as np
from scipy.special import expit
from scipy.stats import norm
import torch
from mechanism import history,fit,predict,values_and_jacobian,sample,counts_from,advance,design

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
sys.path.insert(0,str(ROOT.parent))
from simulator import transition as original_transition


def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')


def reference(qs,x0,cfg):
    nctx=len(qs);reps=cfg['reference_replicates'];h=cfg['evaluation_horizon'];n=cfg['population']
    q=np.repeat(qs,6*reps,axis=0);actions=np.tile(np.repeat(np.arange(6),reps),nctx)
    counts=counts_from(np.repeat(x0,6*reps,axis=0),n)
    reward=np.zeros(len(counts));rng=np.random.default_rng(cfg['reference_seed'])
    for _ in range(h):
        counts,_=sample(counts,q,actions,rng);reward+=counts@np.arange(6)/n/h
    return reward.reshape(nctx,6,reps)


def verify():
    rng=np.random.default_rng(59);x=rng.dirichlet(np.ones(6),20)
    alpha=rng.uniform(-1,1,20);dc=rng.uniform(0,5,20);a=rng.integers(0,6,20)
    q=expit(alpha[:,None]*(np.arange(5)-dc[:,None]))
    ref=np.einsum('bi,bij->bj',x,original_transition(x,np.column_stack((alpha,dc)),a))
    err=float(abs(ref-advance(x,q,a)).max());assert err<1e-12
    value,grad=values_and_jacobian(q[0],x[0],10)
    finite=np.column_stack([(predict(q[0]+np.eye(5)[d]*1e-5,x[0],10)-predict(q[0]-np.eye(5)[d]*1e-5,x[0],10))/2e-5 for d in range(5)])
    grad_error=float(abs(finite-grad).max());assert grad_error<1e-5
    # Rich designs identify all five gap responses; a single uniform snapshot does not.
    rich_rank=int(np.linalg.matrix_rank(design(x,0).reshape(-1,5)))
    poor_rank=int(np.linalg.matrix_rank(design(np.ones((1,6))/6,0).reshape(-1,5)))
    assert rich_rank==5 and poor_rank<5
    return dict(original_formula_max_error=err,autograd_finite_difference_max_error=grad_error,
                diverse_population_design_rank=rich_rank,single_uniform_snapshot_rank=poor_rank,status='PASS')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--quick',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);start=time.perf_counter()
    cfg=dict(contexts=48,budgets=[1,4,16],data_seeds=[7,19,31],history_horizon=12,evaluation_horizon=60,
             population=250,reference_replicates=1024,reference_seed=9191,context_seed=5151,
             curve_stress_amplitude=1.25,nominal_familywise_alpha=.05,material_harm=.01)
    if args.quick:cfg.update(contexts=4,budgets=[1,4],data_seeds=[7],reference_replicates=64)
    for p in [OUT,OUT/'figures',ROOT/'data']:p.mkdir(parents=True,exist_ok=True)
    save('config.json',cfg);save('verification.json',verify())
    rng=np.random.default_rng(cfg['context_seed']);c=cfg['contexts']
    alpha=rng.choice([-1,1],c)*rng.uniform(.2,1,c);dc=rng.uniform(0,5,c)
    base=alpha[:,None]*(np.arange(5)-dc[:,None]);sign=rng.choice([-1,1],c)
    # Stress keeps the invariant gap-response mechanism, relaxes only logistic linearity.
    qs={'original':expit(base),'shape_stress':expit(base+sign[:,None]*cfg['curve_stress_amplitude']*np.sin(np.pi*np.arange(5)/2))}
    deploy=np.full((c,6),1/6)
    np.savez_compressed(ROOT/'data'/'evaluation_latents.npz',alpha=alpha,dc=dc,x0=deploy,**qs)
    records=[];raw={};models=[];allfits=[];zcrit=norm.ppf(1-cfg['nominal_familywise_alpha']/(2*5))
    for regime,qtrue in qs.items():
        print('Reference simulation',regime,flush=True)
        samples=reference(qtrue,deploy,cfg);means=samples.mean(2)
        raw[f'{regime}__reference']=samples
        oracle_det=np.array([predict(q,xx,cfg['evaluation_horizon']) for q,xx in zip(qtrue,deploy)])
        raw[f'{regime}__deterministic']=oracle_det
        for seed in cfg['data_seeds']:
            rg=np.random.default_rng(seed+10000);initial=rg.dirichlet(np.ones(6)*2,(c,max(cfg['budgets'])))
            logs=history(np.repeat(qtrue,max(cfg['budgets']),axis=0),initial.reshape(-1,6),
                         cfg['history_horizon'],cfg['population'],seed+1000)
            logs={k:v.reshape(c,max(cfg['budgets']),*v.shape[1:]) for k,v in logs.items()}
            np.savez_compressed(ROOT/'data'/f'no_ai_{regime}_{seed}.npz',**logs)
            for budget in cfg['budgets']:
                print('Fitting',regime,seed,budget,flush=True)
                for context in range(c):
                    observed={k:v[context,:budget] for k,v in logs.items()}
                    for kind in ['logistic','flexible']:
                        estimate=fit(observed,kind)
                        qhat=estimate['q'];v,g=values_and_jacobian(qhat,deploy[context],cfg['evaluation_horizon'])
                        gd=g-g[0];se=np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',gd,estimate['cov'],gd)))
                        gain=v-v[0];lower=gain-zcrit*se
                        plugin=int(v.argmax());conservative=int(lower.argmax()) if lower.max()>0 else 0
                        actual=means[context];truegain=actual-actual[0]
                        prefix=f'{regime}_{seed}_{budget}_{context}_{kind}'
                        raw[prefix+'__q']=qhat;raw[prefix+'__value']=v;raw[prefix+'__gain_se']=se
                        raw[prefix+'__jacobian']=g;raw[prefix+'__cov']=estimate['cov']
                        models.append(dict(regime=regime,seed=seed,budget=budget,context=context,model=kind,
                                           q=qhat.tolist(),success=estimate['success'],rank=estimate['rank'],
                                           information_condition=estimate['condition'],boundary=estimate['boundary']))
                        for rule,selected in [('plugin',plugin),('conservative',conservative)]:
                            records.append(dict(regime=regime,seed=seed,budget=budget,context=context,
                                model=kind,rule=rule,selected=selected,regret=float(actual.max()-actual[selected]),
                                gain=float(truegain[selected]),material_harm=int(truegain[selected]<-cfg['material_harm']),
                                near_optimal=int(actual.max()-actual[selected]<=.02),
                                q_rmse=float(np.sqrt(np.mean((qhat-qtrue[context])**2))),
                                value_mae=float(abs(v-actual).mean()),
                                simultaneous_gain_coverage=int(np.all(abs(gain[1:]-truegain[1:])<=zcrit*se[1:])),
                                mean_gain_interval_width=float((2*zcrit*se[1:]).mean()),
                                no_ai_regret=float(actual.max()-actual[0]),
                                fit_success=int(estimate['success']),information_rank=estimate['rank'],
                                boundary=int(estimate['boundary'])))
    np.savez_compressed(OUT/'estimates_and_reference.npz',**raw)
    save('estimates.json',models);save('records.json',records)
    with (OUT/'records.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    # Paired differences resample game contexts, averaging the three independent logs first.
    summaries=[];comparisons=[]
    for regime in qs:
        for budget in cfg['budgets']:
            for kind in ['logistic','flexible']:
                for rule in ['plugin','conservative']:
                    subset=[r for r in records if (r['regime'],r['budget'],r['model'],r['rule'])==(regime,budget,kind,rule)]
                    row=dict(regime=regime,budget=budget,model=kind,rule=rule)
                    for metric in ['regret','gain','material_harm','near_optimal','q_rmse','value_mae',
                                   'simultaneous_gain_coverage','mean_gain_interval_width','no_ai_regret','fit_success','boundary']:
                        row[metric]=float(np.mean([r[metric] for r in subset]))
                    summaries.append(row)
            for rule in ['plugin','conservative']:
                diff=[]
                for context in range(c):
                    a=[r['regret'] for r in records if (r['regime'],r['budget'],r['model'],r['rule'],r['context'])==(regime,budget,'flexible',rule,context)]
                    b=[r['regret'] for r in records if (r['regime'],r['budget'],r['model'],r['rule'],r['context'])==(regime,budget,'logistic',rule,context)]
                    diff.append(np.mean(a)-np.mean(b))
                diff=np.array(diff);boot=np.random.default_rng(883).choice(diff,(4000,c)).mean(1)
                comparisons.append(dict(regime=regime,budget=budget,rule=rule,contrast='flexible minus logistic regret',
                    mean=float(diff.mean()),ci95=np.quantile(boot,[.025,.975]).tolist()))
    save('summary.json',summaries);save('paired_comparisons.json',comparisons)
    save('manifest.json',dict(runtime_seconds=time.perf_counter()-start,torch=torch.__version__,numpy=np.__version__,
        estimated_fits=len(models),records=len(records),all_logs_no_ai=True,latent_parameters_not_estimator_inputs=True,
        all_fits_success=all(m['success'] for m in models),rank_deficient_fits=sum(m['rank']<5 for m in models),
        reference_max_se={r:float(raw[f'{r}__reference'].std(2,ddof=1).max()/np.sqrt(cfg['reference_replicates'])) for r in qs},
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}))
    print('COMPLETE',time.perf_counter()-start,flush=True)


if __name__=='__main__':main()
