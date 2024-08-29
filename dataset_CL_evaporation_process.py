import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
from control_torch import drss, forced_response, tf2ss, c2d, perturb_matrices, set_seed, drss_matrices
from signals_torch import steps_sequence
from models import SimpleModel
from evaporation_process import problem_data

class EvaporationDataset(Dataset):
    def __init__(self, seq_len, nx=2, nu=1, ny=2, seed=42, ts=1, data_perturb_percentage = 0, batch_size = 1):
        self.data_perturb_percentage = data_perturb_percentage
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
        self.nx = nx
        self.nu = nu
        self.ny = ny
        # define settings
        self.ts = ts
        self.T = seq_len * ts

        set_seed(42)
        # define nominal model
        self.prob_data = problem_data(self.data_perturb_percentage)

        # define reference
        self.r = torch.zeros((self.seq_len, 2), device=self.device, dtype=torch.float32)
        self.r[:,0] = steps_sequence(self.T, self.ts, 20, 25, 20, 50).T
        self.r[:,1] = steps_sequence(self.T, self.ts, 40, 60, 20, 50).T
        # self.r = torch.tensor([[25.0] * self.seq_len, [49.743] * self.seq_len], device=self.device, dtype=torch.float32).T

    def __len__(self):
        return self.batch_size

    def __getitem__(self, index):
        data = self.prob_data

        x1 = torch.tensor([25], device=self.device, dtype=torch.float32)
        x2 = torch.tensor([49.743], device=self.device, dtype=torch.float32)

        # Desired response
        y_d1 = self.r[:,0].reshape(-1,1)
        y_d2 = self.r[:,1].reshape(-1,1)
        y_d = torch.cat((y_d1, y_d2), dim=1).to(self.device)

        #THIS PART NEEDS WORKING ON, NOT FINISHED
        return data, self.r, y_d


if __name__ == "__main__":
    # Parameters
    batch_size = 1
    # Create dataset and dataloader
    dataset = EvaporationDataset(seq_len=100, ts=1, seed=42, data_perturb_percentage=0)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # print(batch_y_d.shape)
    # print(batch_r.shape)

    ts = 1
    seq_len = 100
    #T = batch_r.shape[1] * ts  # ts*self.seq_len# * 2
    T = seq_len * ts
    t = np.arange(0, T, ts)

    plt.figure(figsize=(10, 6))
    #print(batch_y_d[0, 0:40, 1].cpu().numpy())
    for i in range(batch_size):
        batch_data, batch_r, batch_y_d = next(iter(dataloader))

        print(batch_r.shape)

        print(batch_data)
        plt.plot(t, batch_r[i,:, 0].cpu().numpy(), c='tab:blue', alpha=0.5, label='$r_1$' if i == 0 else "")
        plt.plot(t, batch_r[i,:, 1].cpu().numpy(), c='tab:red', alpha=0.5, label='$r_2$' if i == 0 else "")
        plt.plot(t, batch_y_d[i, :, 0].cpu().numpy(), c='tab:blue', alpha=0.7, label='$y_{d}1$' if i == 0 else "")
        plt.plot(t, batch_y_d[i, :, 1].cpu().numpy(), c='tab:red', alpha=0.7, label='$y_{d}2$' if i == 0 else "")

    plt.ylabel("$y$")
    plt.xlabel("$t$ [s]")
    plt.legend()
    plt.title("Plot of r and y_d")
    plt.show()