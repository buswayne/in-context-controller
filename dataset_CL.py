import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
from control_torch import drss, forced_response, tf2ss, c2d, perturb_matrices, set_seed, perturb_parameters, drss_matrices
from signals_torch import steps_sequence
from models import SimpleModel

class LinearCLDataset(Dataset):
    def __init__(self, seq_len, nx=2, nu=1, ny=1, seed=42, ts=0.01, perturb_percentage = 0, batch_size = 1):
        self.perturb_percentage = perturb_percentage
        self.batch_size = batch_size
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
        #self.G_0 = drss(self.nx, self.nu, self.ny, device=self.device, stricly_proper=True)
        #print(self.G_0)
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
        return self.batch_size

    def __getitem__(self, index):
        # Generate data on-the-fly
        #G = perturb_matrices(*self.G_0, percentage=self.perturb_percentage, device=self.device)
        #G = drss_matrices( 2,1,1)
        G = drss(self.nx, self.nu, self.ny, device=self.device, stricly_proper=True)
        # define reference
        min_val = -10  # Minimum step value
        max_val = 10  # Maximum step value
        min_duration = 0.5  # Minimum duration for each step in seconds
        max_duration = 2  # Maximum duration for each step in seconds
        self.r = steps_sequence(self.T, self.ts, min_val, max_val, min_duration, max_duration, self.device)

        # Desired response
        y_d = forced_response(*self.M, self.r)
        return G, self.r, y_d




class whCLDataset(Dataset):
    def __init__(self, seq_len, nx=2, nu=1, ny=1, seed=42, ts=0.01, perturb_percentage = 0, batch_size =1):
        set_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.perturb_percentage = perturb_percentage
        self.batch_size = batch_size

        self.seq_len = seq_len
        self.nx = nx
        self.nu = nu
        self.ny = ny

        # define settings
        self.ts = ts
        self.T = seq_len * ts

        # define nominal model
        self.G1_0 = drss(self.nx, self.nu, self.ny, device=self.device, stricly_proper=True)
        self.G2_0 = drss(self.nx, self.nu, self.ny, device=self.device, stricly_proper=True)
        n_in = 1
        n_out = 1
        n_hidden = 32
        self.w1 = torch.randn(n_hidden, n_in, device=self.device) / torch.sqrt(
            torch.tensor(n_in, dtype=torch.float32, device=self.device)) * (5 / 3)
        self.b1 = torch.randn(1, n_hidden, device=self.device) * 1.0
        self.w2 = torch.randn(n_out, n_hidden, device=self.device) / torch.sqrt(
            torch.tensor(n_hidden, dtype=torch.float32, device=self.device))
        self.b2 = torch.randn(1, n_out, device=self.device) * 1.0

        # define model reference
        tau = 1
        M_num = torch.tensor([0.01, 1], device=self.device, dtype=torch.float32)  # Numerator coefficients
        M_den = torch.tensor([tau/4, 1], device=self.device, dtype=torch.float32)  # Denominator coefficients
        M = tf2ss(M_num, M_den, device=self.device)  # M
        self.M = c2d(*M, self.ts, device=self.device)


    def __len__(self):
        return self.batch_size

    def __getitem__(self, index):
        # Generate data on-the-fly
        G1 = perturb_matrices(*self.G1_0, percentage=self.perturb_percentage, device=self.device)
        G2 = perturb_matrices(*self.G2_0, percentage=self.perturb_percentage, device=self.device)

        w1_perturbed = perturb_parameters(self.w1, self.perturb_percentage)
        b1_perturbed = perturb_parameters(self.b1, self.perturb_percentage)
        w2_perturbed = perturb_parameters(self.w2, self.perturb_percentage)
        b2_perturbed = perturb_parameters(self.b2, self.perturb_percentage)
        nn_params = {
        "w1" : w1_perturbed,
        "b1" : b1_perturbed,
        "w2" : w2_perturbed,
        "b2" : b2_perturbed
        }
        # define reference
        min_val = -10  # Minimum step value
        max_val = 10  # Maximum step value
        min_duration = 0.5  # Minimum duration for each step in seconds
        max_duration = 2  # Maximum duration for each step in seconds
        self.r = steps_sequence(self.T, self.ts, min_val, max_val, min_duration, max_duration, self.device)
        # Desired response
        y_d = forced_response(*self.M, self.r)

        return G1, G2, nn_params, self.r, y_d


if __name__ == "__main__":
    # Parameters

    batch_size = 32
    # Create dataset and dataloader
    dataset = LinearCLDataset(seq_len=500, ts=0.01, seed=42, perturb_percentage=50)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    batch_G, batch_r, batch_y_d = next(iter(dataloader))

    print(batch_r.shape)
    print(batch_y_d.shape)
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



    """""
    ## PLOT STEP RESPONSE TO ENSURE STABILITY
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    u = torch.full((500,), 10, device=device, dtype=torch.float32)
    u = u.reshape(-1, 1)
    u[:50] = 0
    Y = torch.empty((32, 500), device=device, dtype=torch.float32)
    x_i = torch.zeros((32, 2), device=device, dtype=torch.float32)
    for k in range(32):
        y = forced_response(batch_G[0][k], batch_G[1][k], batch_G[2][k], batch_G[3][k], u, x0=x_i[k])
        if y.dim() == 2 and y.size(1) == 1:
            y = y.squeeze(1)  # Remove the second dimension if it has size 1
        Y[k] = y

    plt.figure(figsize=(20, 10))  # Set the figure size
    for i in range(Y.shape[0]):
        plt.plot(Y[i].cpu().numpy())

    plt.xlabel('Time Step')  # X-axis label
    plt.ylabel('Response Value')  # Y-axis label
    plt.title('Responses of Y[i]')  # Title of the plot
    plt.legend()  # Show legend
    plt.grid(True)  # Add grid for better readability
    plt.show()  # Display the plo
    """""
    




    """""
    
    dataset = whCLDataset(seq_len=500, ts=0.01, seed=42)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    batch_G1, batch_G2, nn_params, batch_r, batch_y_d = next(iter(dataloader))

    print(nn_params)
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
    """""

