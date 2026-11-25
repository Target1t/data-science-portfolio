"""Independent consistency checks for the delivered experiment artifacts."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from models import Dynamics

root=Path(__file__).resolve().parent;out=root/'results'
cfg=json.loads((out/'config.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
metrics=json.loads((out/'metrics.json').read_text())
assert len(metrics)==45
assert hashlib.sha256((root/'provenance'/'cswim_solve.m').read_bytes()).hexdigest()==manifest['original_solve_sha256']
if (root.parent/'solve.m').exists():
    assert hashlib.sha256((root.parent/'solve.m').read_bytes()).hexdigest()==manifest['original_solve_sha256']
sets=[]
for name in ['train','validation','test_iid_static']:
    z=np.load(root/'data'/f'{name}.npz');sets.append(set(map(tuple,z['params'].tolist())))
    assert np.allclose(z['states'].sum(2),1,atol=1e-6)
    assert np.all(z['flows'].sum((2,3))==cfg['population'])
    assert np.allclose(z['flows'].sum(3)/cfg['population'],z['states'][:,:-1],atol=1e-7)
    assert np.allclose(z['flows'].sum(2)/cfg['population'],z['states'][:,1:],atol=1e-7)
assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
max_mass_error=0.
refs=np.load(out/'mc_reference.npz')
for row in metrics:
    data=np.load(out/'predictions'/f'{row["model"]}_{row["seed"]}.npz')
    pred=data[f'{row["split"]}__pred']
    assert np.isfinite(pred).all() and pred.min()>=0
    max_mass_error=max(max_mass_error,float(abs(pred.sum(2)-1).max()))
    assert np.allclose(pred.sum(2),1,atol=1e-5)
    assert abs(data[f'{row["split"]}__tv'][:,1:].mean()-row['rollout_tv'])<1e-7
    if 'mean_regret' in row:
        values=refs[row['split']].mean(2);choice=data[f'{row["split"]}__selected']
        regret=values.max(1)-values[np.arange(len(values)),choice]
        assert np.allclose(regret,data[f'{row["split"]}__regret'])
        assert abs(regret.mean()-row['mean_regret'])<1e-10
z=np.load(root/'data'/'test_initial_shift.npz')
x=torch.tensor(z['states'][:,:-1].reshape(-1,6)[:1000],dtype=torch.float32)
p=torch.tensor(z['params'][:,None,:].repeat(60,axis=1).reshape(-1,2)[:1000],dtype=torch.float32)
a=torch.tensor(z['actions'].reshape(-1)[:1000],dtype=torch.float32)
for kind in ['no_feedback','structured','structured_rollout']:
    model=Dynamics(kind)
    model.load_state_dict(torch.load(out/'models'/f'{kind}_11.pt',weights_only=True)['state_dict'])
    with torch.no_grad():mat=model.kernel(x,p,a).numpy()
    allowed=np.zeros((6,6),bool);allowed[0,:2]=True;allowed[5,4]=True
    for i in range(1,5):allowed[i,[i-1,i+1]]=True
    assert np.max(abs(mat[:,~allowed]))==0
    assert np.allclose(mat.sum(2),1)
result=dict(status='PASS',metric_rows=45,checkpoint_count=len(list((out/'models').glob('*.pt'))),
            max_prediction_mass_error=max_mass_error,original_solver_unchanged=True,
            disjoint_parameter_splits=True,flow_consistency=True,regret_recalculation=True,
            structured_transition_support=True)
(out/'artifact_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
