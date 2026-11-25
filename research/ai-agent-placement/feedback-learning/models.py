"""Neural learners never call the ground-truth transition function."""
import torch
from torch import nn
from simulator import EPSILON


class Dynamics(nn.Module):
    def __init__(self, kind, width=64):
        super().__init__()
        self.kind = kind
        in_dim = 4 if kind == 'no_feedback' else 10
        out_dim = 6 if kind == 'direct' else 24 if kind == 'dense_kernel' else 4
        self.net = nn.Sequential(nn.Linear(in_dim,width),nn.SiLU(),
                                 nn.Linear(width,width),nn.SiLU(),nn.Linear(width,out_dim))

    def features(self, x, params, action):
        z = torch.stack((params[:,0],params[:,1]/5,action/5,(action>0).float()),1)
        return z if self.kind == 'no_feedback' else torch.cat((x,z),1)

    def kernel(self, x, params, action):
        raw = self.net(self.features(x,params,action))
        b = len(x)
        mat = x.new_zeros(b,6,6)
        mat[:,0,0] = 1-EPSILON; mat[:,0,1] = EPSILON; mat[:,5,4] = 1
        if self.kind == 'dense_kernel':
            mat[:,1:5,:] = raw.reshape(b,4,6).softmax(2)
        else:
            down = raw.sigmoid()
            for h in range(1,5):
                mat[:,h,h-1] = down[:,h-1]
                mat[:,h,h+1] = 1-down[:,h-1]
        return mat

    def forward(self, x, params, action):
        if self.kind == 'direct':
            return self.net(self.features(x,params,action)).softmax(1)
        return torch.bmm(x[:,None,:],self.kernel(x,params,action)).squeeze(1)

    def data_loss(self,x,params,action,flows,next_x,n):
        if self.kind == 'direct':
            # Population-level multinomial cross entropy; no individual transition supervision.
            pred = self(x,params,action)
            return -(next_x*pred.clamp_min(1e-8).log()).sum(1).mean()
        mat = self.kernel(x,params,action)
        return -(flows/n*mat.clamp_min(1e-8).log()).sum((1,2)).mean()


def rollout(model,x,params,actions):
    states = [x]
    for t in range(actions.shape[1]):
        x = model(x,params,actions[:,t])
        states.append(x)
    return torch.stack(states,1)
