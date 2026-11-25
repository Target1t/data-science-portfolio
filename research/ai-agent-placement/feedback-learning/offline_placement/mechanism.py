"""Learn and transport a shared gap-response mechanism from no-AI logs.

No estimator takes latent alpha, dc, a true response curve, or AI outcomes.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

EPS=.02
R=.3
GAP=np.abs(np.arange(1,5)[:,None]-np.arange(1,6)[None,:])
GAP_MASK=np.eye(5)[GAP]
D2=np.diff(np.eye(5),n=2,axis=0)


def design(x,action):
    x=np.atleast_2d(x);action=np.broadcast_to(np.asarray(action,int),(len(x),))
    pool=x[:,1:].copy();active=action>0
    pool[np.where(active)[0],action[active]-1]+=R
    den=pool.sum(1,keepdims=True)
    w=np.divide(pool,den,out=np.full_like(pool,.2),where=den>1e-12)
    return np.einsum('bj,ijd->bid',w,GAP_MASK)


def advance(x,q,action):
    x=np.atleast_2d(x);q=np.broadcast_to(q,(len(x),5))
    p=np.einsum('bid,bd->bi',design(x,action),q)
    nxt=np.zeros_like(x,dtype=float)
    nxt[:,0]=x[:,0]*(1-EPS);nxt[:,1]=x[:,0]*EPS;nxt[:,4]=x[:,5]
    for h in range(1,5):
        nxt[:,h-1]+=x[:,h]*p[:,h-1]
        nxt[:,h+1]+=x[:,h]*(1-p[:,h-1])
    return nxt


def sample(counts,q,action,rng):
    p=np.einsum('bid,bd->bi',design(counts/counts.sum(1,keepdims=True),action),np.broadcast_to(q,(len(counts),5)))
    down=rng.binomial(counts[:,1:5],np.clip(p,0,1))
    nxt=np.zeros_like(counts)
    new=rng.binomial(counts[:,0],EPS)
    nxt[:,0]=counts[:,0]-new;nxt[:,1]=new;nxt[:,4]=counts[:,5]
    for h in range(1,5):
        nxt[:,h-1]+=down[:,h-1];nxt[:,h+1]+=counts[:,h]-down[:,h-1]
    return nxt,down


def counts_from(x,n):
    counts=np.floor(x*n).astype(int)
    for i in range(len(x)):
        counts[i,np.argsort(-(x[i]*n-counts[i]))[:n-counts[i].sum()]]+=1
    return counts


def history(q,x0,horizon,n,seed):
    rng=np.random.default_rng(seed);c=counts_from(x0,n);xs=[];downs=[];origins=[]
    for t in range(horizon):
        xs.append(c/n);origins.append(c[:,1:5].copy())
        c,d=sample(c,q,0,rng);downs.append(d)
    return dict(x=np.stack(xs,1),down=np.stack(downs,1),origin=np.stack(origins,1),
                action=np.zeros((len(x0),horizon),int))


def fit(log,kind):
    assert np.all(log['action']==0),'Estimation must use no-AI logs only.'
    w=design(log['x'].reshape(-1,6),0).reshape(-1,5)
    down=log['down'].reshape(-1);n=log['origin'].reshape(-1)
    mask=n>0;w=w[mask];down=down[mask];n=n[mask];total=n.sum()
    def loss_grad_q(q):
        p=np.clip(w@q,1e-10,1-1e-10)
        loss=-np.sum(down*np.log(p)+(n-down)*np.log1p(-p))/total
        grad=w.T@((n*p-down)/(p*(1-p)))/total
        return loss,grad
    if kind=='logistic':
        z=np.column_stack((np.ones(5),np.arange(5)))
        def obj(theta):
            q=expit(z@theta);l,g=loss_grad_q(q)
            return l,z.T@(g*q*(1-q))
        runs=[minimize(obj,np.array(start,float),jac=True,method='L-BFGS-B',bounds=[(-12,12),(-4,4)],
                       options={'ftol':1e-12,'gtol':1e-8,'maxiter':200}) for start in [[0,0],[1,-1],[-1,1]]]
        result=min(runs,key=lambda z:z.fun);q=expit(z@result.x)
        jac=q[:,None]*(1-q[:,None])*z
    else:
        # Convex binomial mixture likelihood: no sigmoid shape assumed for q(d).
        result=minimize(loss_grad_q,np.full(5,.5),jac=True,method='L-BFGS-B',
                        bounds=[(1e-5,1-1e-5)]*5,options={'ftol':1e-12,'gtol':1e-8,'maxiter':300})
        q=result.x;jac=np.eye(5)
    p=np.clip(w@q,1e-8,1-1e-8)
    info=w.T@((n/(p*(1-p)))[:,None]*w)
    cov=jac@np.linalg.pinv(jac.T@info@jac,rcond=1e-10)@jac.T
    evals=np.linalg.eigvalsh(info)
    rank=int(np.linalg.matrix_rank(w*np.sqrt(n[:,None])))
    return dict(q=q,cov=cov,success=bool(result.success),nll=float(result.fun),
                rank=rank,min_information=float(evals[0]),condition=float(evals[-1]/max(evals[0],1e-12)),
                boundary=bool(np.any((q<2e-5)|(q>1-2e-5))))


def predict(q,x0,horizon):
    x=np.repeat(np.asarray(x0)[None,:],6,axis=0);actions=np.arange(6);reward=np.zeros(6)
    for _ in range(horizon):
        x=advance(x,q,actions);reward+=x@np.arange(6)/horizon
    return reward


def values_and_jacobian(q,x0,horizon):
    """Differentiate all six finite-horizon interventions through mean-field dynamics."""
    import torch
    q=torch.tensor(q,dtype=torch.float64,requires_grad=True)
    mask=torch.tensor(GAP_MASK,dtype=torch.float64)
    ai=torch.eye(6,dtype=torch.float64)[:,1:]*R
    def values(q):
        x=torch.tensor(x0,dtype=torch.float64)[None,:].repeat(6,1)
        value=x.new_zeros(6)
        for _ in range(horizon):
            pool=x[:,1:]+ai;w=pool/pool.sum(1,keepdim=True).clamp_min(1e-12)
            p=torch.einsum('bj,ijd,d->bi',w,mask,q)
            cols=[x[:,0]*(1-EPS),x[:,0]*EPS,x.new_zeros(6),x.new_zeros(6),x[:,5],x.new_zeros(6)]
            for h in range(1,5):
                cols[h-1]=cols[h-1]+x[:,h]*p[:,h-1]
                cols[h+1]=cols[h+1]+x[:,h]*(1-p[:,h-1])
            x=torch.stack(cols,1);value=value+x@torch.arange(6,dtype=x.dtype)/horizon
        return value
    value=values(q)
    grads=torch.stack([torch.autograd.grad(value[k],q,retain_graph=k<5)[0] for k in range(6)])
    return value.detach().numpy(),grads.detach().numpy()
