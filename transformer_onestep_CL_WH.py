import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformer_onestep import GPT, GPTConfig
from control_torch import forced_response
from models import SimpleModel

class GPTClosedLoop(nn.Module):
    def __init__(self, gptconf):
        super().__init__()
        self.gpt_model = GPT(gptconf)
        self.nx = gptconf.n_x

    def forward(self, G1, G2, nn_params, r):

        device = r.device
        w1 = nn_params['w1']
        b1 = nn_params['b1']
        w2 = nn_params['w2']
        b2 = nn_params['b2']
        def nn_fun(x,w1,b1,w2,b2):
            out = x @ w1.T + b1
            out = torch.tanh(out)
            out = out @ w2.T + b2
            return out

        b, t, nr = r.size()

        E = torch.empty_like(r, device=device, dtype=torch.float32)
        U = torch.empty((b, t+1, nr), device=device, dtype=torch.float32)
        Y1 = torch.empty_like(r, device=device, dtype=torch.float32)
        Y2 = torch.empty_like(r, device=device, dtype=torch.float32)
        Y = torch.empty_like(r, device=device, dtype=torch.float32)

        # Initial conditions
        U[:, 0, 0] = 0
        y1_i = torch.zeros((b, 1), device=device, dtype=torch.float32)
        y2_i = torch.zeros((b, 1), device=device, dtype=torch.float32)
        y_i = torch.zeros((b, 1), device=device, dtype=torch.float32)
        x1_i = torch.zeros((b, self.nx), device=device, dtype=torch.float32)
        x2_i = torch.zeros((b, self.nx), device=device, dtype=torch.float32)

        for i in range(t):
            # Print current time instant
            # print('time instant:', i)

            # Compute error
            e_i = r[:, i, :] - y_i
            Y1[:,i,:] = y1_i
            Y2[:, i, :] = y2_i
            Y[:, i, :] = y_i
            E[:, i, :] = e_i

            # Controller u(t) = C(e(t), u(t-1))
            pred = self.gpt_model(E[:, :i + 1, :], U[:, :i + 1, :])

            U[:, i + 1, :] = pred[:, -1, :]  # Just for coherence

            # Simulate system response I NEED TO CHECK AND PUT IN ORDER
            for k in range(b):
                y1, x1_i[k] = forced_response(G1[0][k],G1[1][k],G1[2][k],G1[3][k], U[0, i:i + 2, :], return_x=True, x0=x1_i[k])
                y1_i[k] = y1[-1]
                #y1_i[k] = y1
            for k in range(b):
                reshaped_y1_i = y1_i[k].reshape(-1, 1)
                y2 = nn_fun(reshaped_y1_i,w1[0,:,:],b1[0,:,:],w2[0,:,:],b2[0,:,:])
                #y2_i[k] = y2[-1]
                y2_i[k] = y2
            for k in range(b):
                reshaped_y2_i = y2_i[k].reshape(-1, 1)
                y3, x2_i[k] = forced_response(G2[0][k], G2[1][k], G2[2][k], G2[3][k], reshaped_y2_i, return_x=True, x0=x2_i[k])
                y_i[k] = y3[-1]

        return Y

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer


