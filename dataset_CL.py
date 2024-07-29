import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
from control_torch import drss, forced_response, tf2ss, c2d, perturb_matrices, set_seed
from signals_torch import steps_sequence

class LinearCLDataset(Dataset):
    def __init__(self, seq_len, nx=2, nu=1, ny=1, seed=42, ts=0.01):
        set_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seq_len = seq_len
        self.nx = nx
        self.nu = nu
        self.ny = ny

        # define settings
        self.ts = ts
        self.T = seq_len * ts

        # define nominal model
        self.G_0 = drss(self.nx, self.nu, self.ny, device=self.device, stricly_proper=True)

        # define model reference
        tau = 1
        M_num = torch.tensor([0.01, 1], device=self.device, dtype=torch.float32)  # Numerator coefficients
        M_den = torch.tensor([tau/4, 1], device=self.device, dtype=torch.float32)  # Denominator coefficients
        M = tf2ss(M_num, M_den, device=self.device)  # M
        self.M = c2d(*M, self.ts, device=self.device)

        # define reference
        min_val = -10  # Minimum step value
        max_val = 10  # Maximum step value
        min_duration = 0.5  # Minimum duration for each step in seconds
        max_duration = 2  # Maximum duration for each step in seconds
        # self.r = steps_sequence(self.T, self.ts, min_val, max_val, min_duration, max_duration, self.device)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # Generate data on-the-fly
        G = perturb_matrices(*self.G_0, percentage=0, device=self.device)

        # define reference
        min_val = -10  # Minimum step value
        max_val = 10  # Maximum step value
        min_duration = 0.5  # Minimum duration for each step in seconds
        max_duration = 2  # Maximum duration for each step in seconds
        self.r = steps_sequence(self.T, self.ts, min_val, max_val, min_duration, max_duration, self.device)

        # Desired response
        y_d = forced_response(*self.M, self.r)

        return G, self.r, y_d

if __name__ == "__main__":
    # Parameters
    batch_size = 32

    # Create dataset and dataloader
    dataset = LinearCLDataset(seq_len=500, ts=0.01, seed=42)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    batch_G, batch_r, batch_y_d = next(iter(dataloader))

    # print(batch_r.shape, batch_y_d.shape)

    ts = 1e-2
    T = batch_r.shape[1] * ts  # ts*self.seq_len# * 2
    t = np.arange(0, T, ts)

    for i in range(0, batch_r.shape[0]):
        plt.subplot(111)
        plt.plot(t, batch_r[i, :, 0].cpu(), c='tab:blue', alpha=0.2, label='$r$')
        plt.plot(t, batch_y_d[i, :, 0].cpu(), c='tab:red', alpha=0.2, label='$y_d$')
        plt.ylabel("$y_L$")
        plt.xlabel("$t$ [s]")
    plt.show()
