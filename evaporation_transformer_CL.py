import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformer_onestep import GPT, GPTConfig
from control_torch import forced_response
from evaporation_process import dynamics, intermediate_vars

class GPTClosedLoop(nn.Module):
    def __init__(self, gptconf):
        super().__init__()
        self.gpt_model = GPT(gptconf)
        self.nx = gptconf.n_x
        self.ny = gptconf.n_y

    def forward(self, data, r):
        Ts =1
        device = r.device
        b, t, nr = r.size() # 1,500,2

        E = torch.empty_like(r, device=device, dtype=torch.float32)
        U = torch.empty((b, t+1, nr), device=device, dtype=torch.float32)
        Y = torch.empty_like(r, device=device, dtype=torch.float32)

        # Initial conditions
        U[:, 0, :] = 0
        y_i = torch.zeros((b, self.ny), device=device, dtype=torch.float32)
        x_i = torch.zeros((b, self.nx), device=device, dtype=torch.float32)
        #dovrei mettere uno stato iniziale (e anche uscita ) diverso da 0

        for i in range(t):
            # Print current time instant
            #print('time instant:', i)

            # Compute error
            e_i = r[:, i, :] - y_i
            Y[:, i, :] = y_i
            E[:, i, :] = e_i

            # Controller u(t) = C(e(t), u(t-1))
            pred = self.gpt_model(E[:, :i + 1, :], U[:, :i + 1, :])

            U[:, i + 1, :] = pred[:, -1, :]  # Just for coherence

            # Simulate system response
            for k in range(b):
                # noiseless case
                a = U[k,i+1, :]  # action
                # store the nominal value
                s = x_i[k]
                #print(s.shape) #2
                intermediate_data = intermediate_vars(s, a, data)
                x_dot = dynamics(s, a, intermediate_data).reshape(-1)
                #print(x_dot.shape)
                # Integrate dynamics using forward Euler integration
                s_next = s + Ts * x_dot

                #print("s_next shape:", s_next.shape)
                #print("y_i shape:", y_i[k].shape)
                y_i = y_i.clone()
                x_i = x_i.clone()
                y_i[k] = s_next
                x_i[k] = s_next

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


