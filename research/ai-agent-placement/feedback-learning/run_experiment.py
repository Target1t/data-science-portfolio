"""Run the complete CSWIM-linked benchmark; no MATLAB/Excel input required."""
from pathlib import Path
import argparse, copy, csv, hashlib, json, platform, time
import numpy as np
import torch
from models import Dynamics, rollout
from simulator import make_split, trajectories, deterministic, sample_step, initial_counts, verify

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'
KINDS = ['no_feedback','direct','dense_kernel','structured','structured_rollout']


def save_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def tensors(data):
    return {k:torch.tensor(data[k],dtype=torch.float32) for k in ['params','actions','states','flows']}


@torch.no_grad()
def validation(model,d,steps=10):
    pred = rollout(model,d['states'][:,0],d['params'],d['actions'][:,:steps])
    return float((pred[:,1:]-d['states'][:,1:steps+1]).abs().sum(2).mean()/2)


def fit(kind,seed,train,val,cfg,initial=None):
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    model = Dynamics(kind)
    if initial is not None: model.load_state_dict(initial)
    opt = torch.optim.AdamW(model.parameters(),lr=.002 if initial is None else .0005,weight_decay=1e-5)
    tr, va = tensors(train), tensors(val)
    ntraj,h = tr['actions'].shape
    x=tr['states'][:,:-1].reshape(-1,6)
    nx=tr['states'][:,1:].reshape(-1,6)
    p=tr['params'][:,None,:].expand(-1,h,-1).reshape(-1,2)
    a=tr['actions'].reshape(-1)
    flow=tr['flows'].reshape(-1,6,6)
    best = validation(model,va); state=copy.deepcopy(model.state_dict()); stale=0; history=[]
    max_epochs=cfg['epochs'] if initial is None else cfg['finetune_epochs']
    start=time.perf_counter()
    for epoch in range(max_epochs):
        model.train(); losses=[]
        if initial is None:
            order=torch.randperm(len(x))
            for ids in order.split(cfg['batch']):
                opt.zero_grad()
                loss=model.data_loss(x[ids],p[ids],a[ids],flow[ids],nx[ids],int(train['n']))
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
                losses.append(float(loss.detach()))
        else:
            # Same observed data, differentiable 8-step windows; no ground-truth formula.
            for _ in range(12):
                ids=torch.tensor(np_rng.integers(0,ntraj,128))
                starts=torch.tensor(np_rng.integers(0,h-8+1,128))
                steps=starts[:,None]+torch.arange(8)[None,:]
                xstart=tr['states'][ids,starts]
                seq_a=tr['actions'][ids[:,None],steps]
                target=tr['states'][ids[:,None],steps+1]
                opt.zero_grad()
                pred=rollout(model,xstart,tr['params'][ids],seq_a)[:,1:]
                multi=(pred-target).square().mean()
                one=model.data_loss(xstart,tr['params'][ids],seq_a[:,0],
                                    tr['flows'][ids,starts],target[:,0],int(train['n']))
                loss=one+10*multi
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
                losses.append(float(loss.detach()))
        model.eval(); v=validation(model,va)
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),validation_tv10=v))
        if v < best-1e-6:
            best=v;state=copy.deepcopy(model.state_dict());stale=0
        else: stale+=1
        if (epoch+1)%25==0: print(f'{kind} seed={seed} epoch={epoch+1} val-TV10={v:.5f}',flush=True)
        if stale>=18: break
    model.load_state_dict(state);model.eval()
    seconds=time.perf_counter()-start
    torch.save(dict(kind=kind,seed=seed,state_dict=state),OUT/'models'/f'{kind}_{seed}.pt')
    save_json(OUT/'models'/f'{kind}_{seed}_history.json',history)
    info=dict(kind=kind,seed=seed,epochs=len(history),best_validation_tv10=best,
              training_seconds=seconds,parameters=sum(p.numel() for p in model.parameters()))
    print('FINISHED',info,flush=True)
    return model,info


def test_suite(cfg):
    base=make_split(cfg['test_scenarios'],cfg['test_horizon'],cfg['population'],cfg['test_seed'])
    rng=np.random.default_rng(cfg['test_seed']+1)
    shift=copy.deepcopy(base)
    initial=rng.dirichlet(np.ones(6)*.2,cfg['test_scenarios']).repeat(6,axis=0)
    shift['states'],shift['flows']=trajectories(shift['params'],initial,shift['actions'],
                                               cfg['population'],cfg['test_seed']+2)
    switch=copy.deepcopy(base)
    switch['actions'][:,cfg['switch_time']:] = (switch['actions'][:,cfg['switch_time']:]+3)%6
    switch['states'],switch['flows']=trajectories(switch['params'],base['states'][:,0],switch['actions'],
                                                 cfg['population'],cfg['test_seed']+3)
    return dict(iid_static=base,initial_shift=shift,ai_switch=switch)


def mc_policy_reference(data,cfg):
    """Independent complete trajectories; no single-step independence assumption."""
    reps=cfg['mc_replicates']; b=len(data['params']); h=cfg['test_horizon']
    params=np.repeat(data['params'],reps,axis=0)
    acts=np.repeat(data['actions'][:,0],reps)
    counts=initial_counts(np.repeat(data['states'][:,0],reps,axis=0),cfg['population'])
    rng=np.random.default_rng(cfg['reference_seed'])
    rewards=np.zeros(len(counts))
    for _ in range(h):
        counts,_=sample_step(counts,params,acts,rng)
        rewards += counts@np.arange(6)/cfg['population']/h
    return rewards.reshape(b,reps).reshape(cfg['test_scenarios'],6,reps)


@torch.no_grad()
def evaluate(model,kind,seed,suites,refs,cfg):
    metrics=[];details={}
    for split,d in suites.items():
        td=tensors(d)
        start=time.perf_counter()
        pred=rollout(model,td['states'][:,0],td['params'],td['actions']).numpy()
        elapsed=time.perf_counter()-start
        actual=d['states'];det=deterministic(d['params'],actual[:,0],d['actions'])
        tv=.5*abs(pred-actual).sum(2)
        det_tv=.5*abs(pred-det).sum(2)
        reward_pred=(pred[:,1:]@np.arange(6)).mean(1)
        true_reward=(actual[:,1:]@np.arange(6)).mean(1)
        # Teacher-forced one-step error on the same test trajectories.
        h=d['actions'].shape[1]
        one=model(td['states'][:,:-1].reshape(-1,6),
                  td['params'][:,None,:].expand(-1,h,-1).reshape(-1,2),td['actions'].reshape(-1))
        one=one.numpy().reshape(len(pred),h,6)
        row=dict(model=kind,seed=seed,split=split,one_step_tv=float(.5*abs(one-actual[:,1:]).sum(2).mean()),
                 rollout_tv=float(tv[:,1:].mean()),final_tv=float(tv[:,-1].mean()),
                 deterministic_rollout_tv=float(det_tv[:,1:].mean()),
                 reward_mae=float(abs(reward_pred-true_reward).mean()),inference_seconds=elapsed)
        # Scenario-level metrics retain all six actions together for paired bootstrap.
        grouped_tv=tv[:,1:].mean(1).reshape(-1,6).mean(1)
        details[split]=dict(pred=pred,tv=tv,scenario_tv=grouped_tv)
        if split in refs:
            samples=refs[split];means=samples.mean(2);best=means.argmax(1)
            selected=reward_pred.reshape(-1,6).argmax(1)
            regret=means.max(1)-means[np.arange(len(means)),selected]
            # Same independently generated MC sample set across model comparisons.
            row.update(mean_regret=float(regret.mean()),p90_regret=float(np.quantile(regret,.9)),
                       exact_best_rate=float((selected==best).mean()),
                       within_002_rate=float((regret<=.02).mean()))
            details[split].update(regret=regret,selected=selected,reward_pred=reward_pred.reshape(-1,6))
        metrics.append(row)
    np.savez_compressed(OUT/'predictions'/f'{kind}_{seed}.npz',
                        **{f'{s}__{k}':v for s,z in details.items() for k,v in z.items()})
    return metrics


def bootstrap_difference(values_a,values_b):
    """Average over paired training seeds, resample independent parameter scenarios."""
    d=np.mean(values_a-values_b,axis=0)
    rng=np.random.default_rng(441)
    boot=d[rng.integers(0,len(d),(4000,len(d)))].mean(1)
    return dict(mean_difference=float(d.mean()),ci95_low=float(np.quantile(boot,.025)),
                ci95_high=float(np.quantile(boot,.975)),unit='parameter scenario; training seeds averaged')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--quick',action='store_true')
    parser.add_argument('--evaluate-only',action='store_true');args=parser.parse_args()
    start=time.perf_counter();torch.set_num_threads(2)
    cfg=dict(train_scenarios=128,validation_scenarios=32,test_scenarios=48,
             train_horizon=32,test_horizon=60,population=1000,batch=1024,
             epochs=100,finetune_epochs=45,seeds=[11,29,47],mc_replicates=1024,
             train_seed=101,validation_seed=202,test_seed=303,reference_seed=404,switch_time=20)
    if args.quick:
        cfg.update(train_scenarios=16,validation_scenarios=8,test_scenarios=8,
                   epochs=3,finetune_epochs=2,seeds=[11],mc_replicates=8)
    for path in [OUT,ROOT/'data',OUT/'models',OUT/'predictions',OUT/'figures']:path.mkdir(parents=True,exist_ok=True)
    save_json(OUT/'config.json',cfg)
    save_json(OUT/'verification.json',verify())
    train=make_split(cfg['train_scenarios'],cfg['train_horizon'],cfg['population'],cfg['train_seed'])
    val=make_split(cfg['validation_scenarios'],cfg['train_horizon'],cfg['population'],cfg['validation_seed'])
    for name,d in [('train',train),('validation',val)]: np.savez_compressed(ROOT/'data'/f'{name}.npz',**d)
    suites=test_suite(cfg)
    for name,d in suites.items():np.savez_compressed(ROOT/'data'/f'test_{name}.npz',**d)
    # Entire (alpha,dc) groups, all actions and time points, stay in one split.
    sets=[set(map(tuple,z['params'].tolist())) for z in [train,val,suites['iid_static']]]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    refs={name:mc_policy_reference(d,cfg) for name,d in suites.items() if name!='ai_switch'}
    np.savez_compressed(OUT/'mc_reference.npz',**refs)
    metrics=[];training=[]
    for seed in cfg['seeds']:
        structured_state=None
        for kind in KINDS:
            if args.evaluate_only:
                model=Dynamics(kind)
                model.load_state_dict(torch.load(OUT/'models'/f'{kind}_{seed}.pt',weights_only=True)['state_dict'])
                model.eval();info=dict(kind=kind,seed=seed,evaluation_only=True)
            else:
                model,info=fit(kind,seed,train,val,cfg,structured_state if kind=='structured_rollout' else None)
            if kind=='structured': structured_state=copy.deepcopy(model.state_dict())
            training.append(info);metrics.extend(evaluate(model,kind,seed,suites,refs,cfg))
    save_json(OUT/'metrics.json',metrics)
    if not args.evaluate_only: save_json(OUT/'training.json',training)
    keys=list(dict.fromkeys(k for row in metrics for k in row))
    with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(metrics)
    comparisons=[]
    for split in suites:
        for baseline in ['no_feedback','direct','dense_kernel','structured']:
            for metric in ['scenario_tv']+(['regret'] if split in refs else []):
                arrays=[]
                for modelname in ['structured_rollout',baseline]:
                    arrays.append(np.stack([np.load(OUT/'predictions'/f'{modelname}_{s}.npz')[f'{split}__{metric}'] for s in cfg['seeds']]))
                comparisons.append(dict(split=split,metric=metric,model='structured_rollout',baseline=baseline,
                                        **bootstrap_difference(*arrays)))
    save_json(OUT/'paired_comparisons.json',comparisons)
    ref_info={k:dict(max_standard_error=float((v.std(2,ddof=1)/np.sqrt(v.shape[2])).max()),
                    mean_standard_error=float((v.std(2,ddof=1)/np.sqrt(v.shape[2])).mean())) for k,v in refs.items()}
    manifest=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                  device='CPU',threads=2,run_mode='evaluate_only' if args.evaluate_only else 'full',
                  execution_seconds=time.perf_counter()-start,
                  training_seconds=sum(z.get('training_seconds',0) for z in json.loads((OUT/'training.json').read_text())),
                  reference_uncertainty=ref_info,
                  original_solve_sha256=hashlib.sha256((ROOT/'provenance'/'cswim_solve.m').read_bytes()).hexdigest(),
                  files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(ROOT.glob('*.py'))},
                  dataset_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').glob('*.npz')})
    save_json(OUT/'manifest.json',manifest)
    print('COMPLETE',manifest['execution_seconds'],flush=True)


if __name__=='__main__':main()
