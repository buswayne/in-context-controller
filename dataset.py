import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
from control_torch import drss, forced_response, tf2ss, c2d, perturb_matrices, set_seed

class LinearDataset(Dataset):
    def __init__(self, seq_len, nx=2, nu=1, ny=1, seed=42, ts=0.01, return_y=False):
        set_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seq_len = seq_len + 1
        self.nx = nx
        self.nu = nu
        self.ny = ny

        # define settings
        self.ts = ts
        self.return_y = return_y

        # define nominal model
        self.G_0 = drss(self.nx, self.nu, self.ny, device=self.device)

        # define model reference
        tau = 1
        M_num = torch.tensor([0.01, 1], device=self.device, dtype=torch.float32)  # Numerator coefficients
        M_den = torch.tensor([tau, 1], device=self.device, dtype=torch.float32)  # Denominator coefficients
        M = tf2ss(M_num, M_den, device=self.device)  # M
        M_inv = tf2ss(M_den, M_num, device=self.device)  # M^-1, num den are inverted
        self.M = c2d(*M, self.ts, device=self.device)
        self.M_inv = c2d(*M_inv, self.ts, device=self.device)

        self.u = 1000*torch.randn(self.seq_len, self.nu, device=self.device, dtype=torch.float32)

    def __len__(self):
        return 32

    def __getitem__(self, index):
        # Generate data on-the-fly
        G = perturb_matrices(*self.G_0, percentage=0, device=self.device)

        # u = 1000*torch.randn(self.seq_len, self.nu, device=self.device, dtype=torch.float32)
        # u = torch.ones(self.seq_len, self.nu, device=device, dtype=torch.float32)
        u = self.u

        # Simulate forced response using custom GPU function
        y = forced_response(*G, u)

        # Prefilter with M
        u_L = forced_response(*self.M, u)
        y_L = forced_response(*self.M, y)

        # Compute virtual reference
        r_v = forced_response(*self.M_inv, y_L)

        # Compute virtual error
        e_v = r_v - y_L

        # Align to have proper signal
        u_L = u_L[:-1] / 200
        y_L = y_L[1:]
        r_v = r_v[1:]
        e_v = e_v[1:] / 200

        if self.return_y:
            return y_L, u_L, e_v
        else:
            return u_L, e_v


if __name__ == "__main__":
    # Parameters
    batch_size = 32

    # Create dataset and dataloader
    dataset = LinearDataset(seq_len=500, ts=0.01, seed=42, return_y=True)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    batch_y, batch_u, batch_e = next(iter(dataloader))

    ts = 1e-2
    T = batch_u.shape[1] * ts  # ts*self.seq_len# * 2
    t = np.arange(0, T, ts)

    for i in range(0, batch_u.shape[0]):
        plt.subplot(311)
        plt.plot(t, batch_u[i, :, 0].cpu(), c='tab:blue', alpha=0.2)
        # plt.legend(['$e_v$'])
        plt.ylabel("$u_L$")
        plt.tick_params('x', labelbottom=False)

        plt.subplot(312)
        plt.plot(t, batch_y[i, :, 0].cpu(), c='tab:blue', alpha=0.2)
        # plt.legend(['$u$'])
        plt.ylabel("$y_L$")
        plt.xlabel("$t$ [s]")

        plt.subplot(313)
        plt.plot(t, batch_e[i, :, 0].cpu(), c='tab:blue', alpha=0.2)
        # plt.legend(['$y$'])
        plt.ylabel("$e_v$")
        plt.tick_params('x', labelbottom=False)

    plt.show()
