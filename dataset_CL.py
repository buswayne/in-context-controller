import numpy as np
import torch
import torch.nn as nn
import time
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
from control_torch import drss, drss_matrices, forced_response, tf2ss, c2d, perturb_matrices, set_seed
from signals_torch import steps_sequence
import pickle as pkl

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['axes.labelsize']=14
plt.rcParams['xtick.labelsize']=11
plt.rcParams['ytick.labelsize']=11
plt.rcParams['axes.grid']=True
plt.rcParams['axes.xmargin']=0

class LinearCLDataset(Dataset):
    def __init__(self, seq_len, nx=2, nu=1, ny=1, seed=42, ts=0.01, random_start=False):
        set_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seq_len = seq_len
        self.nx = nx
        self.nu = nu
        self.ny = ny

        # define settings
        self.ts = ts
        self.T = seq_len * ts
        self.random_start = random_start

        # define nominal model
        self.G_0 = drss_matrices(self.nx, self.nu, self.ny, device=self.device, strictly_proper=True)

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
        self.r = steps_sequence(self.T, self.ts, min_val, max_val, min_duration, max_duration, self.device)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # Generate data on-the-fly
        G = perturb_matrices(*self.G_0, percentage=20, device=self.device)
        # G = drss(self.nx, self.nu, self.ny, device=self.device, strictly_proper=True)
        # define reference
        min_val = -10  # Minimum step value
        max_val = 10  # Maximum step value
        min_duration = 0.5  # Minimum duration for each step in seconds
        max_duration = 2  # Maximum duration for each step in seconds

        self.r = steps_sequence(self.T, self.ts, min_val, max_val, min_duration, max_duration, self.device)

        # Desired response
        y_d = forced_response(*self.M, self.r)

        return G, self.r, y_d


class NonlinearCLDataset(Dataset):
    def __init__(self, seq_len, nx=2, nu=1, ny=1, seed=42, ts=0.01, random_start=False):
        set_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seq_len = seq_len
        self.nx = nx
        self.nu = nu
        self.ny = ny

        # define settings
        self.ts = ts
        self.T = seq_len * ts
        self.random_start = random_start

        # define nominal model

        # define nominal model
        n_in = 1
        n_out = 1
        n_hidden = 32

        # Generate random weights and biases using PyTorch's random functions and move them to the GPU
        self.w1 = torch.randn(n_hidden, n_in, device=self.device) / torch.sqrt(
            torch.tensor(n_in, dtype=torch.float32, device=self.device)) * (5 / 3)
        self.b1 = torch.randn(1, n_hidden, device=self.device) * 1.0
        self.w2 = torch.randn(n_out, n_hidden, device=self.device) / torch.sqrt(
            torch.tensor(n_hidden, dtype=torch.float32, device=self.device))
        self.b2 = torch.randn(1, n_out, device=self.device) * 1.0
        self.G_01 = drss_matrices(self.nx, self.nu, self.ny, device=self.device)
        self.G_02 = drss_matrices(self.nx, self.nu, self.ny, device=self.device)

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

        # A simple ff neural network
        def nn_fun(x):
            out = x @ self.w1.T + self.b1
            out = torch.tanh(out)
            out = out @ self.w2.T + self.b2
            return out
        # Generate data on-the-fly
        G1 = perturb_matrices(*self.G_01, percentage=0, device=self.device)
        G2 = perturb_matrices(*self.G_02, percentage=0, device=self.device)
        # define reference
        min_val = -10  # Minimum step value
        max_val = 10  # Maximum step value
        min_duration = 0.5  # Minimum duration for each step in seconds
        max_duration = 2  # Maximum duration for each step in seconds

        self.r = steps_sequence(self.T, self.ts, min_val, max_val, min_duration, max_duration, self.device)

        # Desired response
        y_d = forced_response(*self.M, self.r)

        return G1, G2, self.r, y_d


if __name__ == "__main__":
    # Parameters
    batch_size = 50

    # Create dataset and dataloader
    dataset = LinearCLDataset(seq_len=500, ts=0.01, seed=42, random_start=False)
    # dataset = NonlinearCLDataset(seq_len=500, ts=0.01, seed=42, random_start=True)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    ts = 1e-2
    T = ts*500# * 2
    t = torch.arange(0, T, ts).view(-1, 1).to("cuda")

    fig = plt.figure(figsize=(5,3))
    u = torch.ones_like(t)

    G_dict = {}

    # for i in range(batch_size):
    #     batch_G, batch_r, batch_y_d = next(iter(dataloader))
    #     print(batch_G)
    #     G_dict[i] = batch_G[0][0], batch_G[1][0], batch_G[2][0], batch_G[3][0]



    # def nn_fun(x):
    #     out = x @ dataset.w1.T + dataset.b1
    #     out = torch.tanh(out)
    #     out = out @ dataset.w2.T + dataset.b2
    #     return out
    #
    # for i in range(batch_size):
    #     batch_G1, batch_G2, batch_r, batch_y_d, batch_y0 = next(iter(dataloader))
    #     y1 = forced_response(batch_G1[0][0],batch_G1[1][0],batch_G1[2][0],batch_G1[3][0], u)
    #     y2 = nn_fun(y1)
    #     y = forced_response(batch_G2[0][0],batch_G2[1][0],batch_G2[2][0],batch_G2[3][0], y1)
    #     plt.plot(t.cpu(), y.cpu())
    # plt.show()


    # ts = 1e-2

    for i in range(batch_size):
        batch_G, batch_r, batch_y_d = next(iter(dataloader))

        T = batch_r.shape[1] * ts  # ts*self.seq_len# * 2
        t = np.arange(0, T, ts)
        y = forced_response(batch_G[0][0], batch_G[1][0], batch_G[2][0], batch_G[3][0], u)

        # plt.subplot(211)
        # plt.plot(t, batch_r[0, :, 0].cpu(), c='k', alpha=0.2, label='$r$')
        # plt.plot(t, batch_y_d[0, :, 0].cpu(), c='tab:red', alpha=0.2, label='$y_d$')
        # plt.ylabel("$y_L$")
        # plt.xlabel("$t$ [s]")

        plt.subplot(111)
        plt.plot(t, u.cpu(), label='$r$', c='k', linewidth=1)
        plt.plot(t, y.cpu(), label='$y$', c='tab:blue', alpha=0.2)
        plt.xlim([0, 3])
        plt.legend(['$u$','$y$'])
        plt.xlabel('$t$ [s]')

    plt.show()
