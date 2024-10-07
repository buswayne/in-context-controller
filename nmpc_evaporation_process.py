import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset_CL_evaporation_process import EvaporationDataset
from evaporation_process import problem_data, dynamics

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import pickle


class NMPC(nn.Module):
    def __init__(self, nx, nu, N, data):
        super(NMPC, self).__init__()
        self.nx = nx
        self.nu = nu
        self.N = N
        self.data = data

        # Initialize the control sequence with parameters for optimization
        # self.u_sequence = nn.Parameter(torch.randn(nu, N, device='cuda:0'))
        # Initialize the control sequence with u1 = 191 and u2 = 200 across the horizon
        self.u_sequence = nn.Parameter(torch.tensor([[191], [200]], dtype=torch.float32, device='cuda').repeat(1, N))

    def forward(self, x0, x_ref):
        """
        Perform the forward pass, which predicts the trajectory and computes the cost.

        Args:
        - x0 (torch.Tensor): Initial state vector (shape: (state_dim,))
        - x_ref (torch.Tensor): Reference state vector (shape: (state_dim,))
        - system_dynamics (function): Function that defines the system dynamics
        - data: Additional data for the system dynamics

        Returns:
        - cost (torch.Tensor): The computed cost
        """
        trajectory = torch.zeros(self.N + 1, self.nx, device=x0.device)
        trajectory[0] = x0

        x = x0

        for i in range(self.N):
            u = self.u_sequence[:, i].view(-1, 1)
            x_dot = dynamics(x, u, *self.data)
            x = x + 1 * x_dot
            trajectory[i + 1] = x

        # Assume trajectory and x_ref are torch tensors of the same shape
        mse_loss = nn.MSELoss()
        cost = mse_loss(trajectory, x_ref)#torch.sum((trajectory - x_ref) ** 2) / ( self.N + 1 )

        return cost

    def optimize_control_sequence(self, x0, x_ref, lr=1, num_iter=1000):
        """
        Optimize the control sequence using gradient descent.

        Args:
        - x0 (torch.Tensor): Initial state vector (shape: (state_dim,))
        - x_ref (torch.Tensor): Reference state vector (shape: (state_dim,))
        - system_dynamics (function): Function that defines the system dynamics
        - data: Additional data for the system dynamics
        - lr (float): Learning rate for the optimizer
        - num_iter (int): Number of optimization iterations

        Returns:
        - optimal_u (torch.Tensor): The optimized control sequence (shape: (control_dim, N))
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for k in range(num_iter):
            optimizer.zero_grad()

            # Forward pass: compute the cost
            cost = self.forward(x0, x_ref)

            # # Print cost and control sequence for debugging
            # print('Iteration:', k, cost)

            # Backward pass: compute the gradients
            cost.backward()

            # Perform a gradient descent step
            optimizer.step()

            if cost <= 1e-4:
                break



        return self.u_sequence[:, 0].detach()  # Return the first control action (MPC principle)

def main():

    torch.cuda.set_device('cuda:3')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define state and control dimensions
    nx = 2
    nu = 2
    # Define simulation parameters
    T = 30
    t = torch.arange(T)
    N = 5  # Prediction horizon

    # Initialize the initial state, reference state, and system parameters
    x0 = torch.tensor([25.0, 49.0], dtype=torch.float32, device=device)
    x_ref = torch.tensor([30.0, 40.0], dtype=torch.float32, device=device)

    # Initialize the model
    dataset = EvaporationDataset(seq_len=100, ts=1, seed=42, data_perturb_percentage=0, device=device)
    dataloader = DataLoader(dataset, batch_size=1)
    _, r, y_d = next(iter(dataloader))

    # Repeat the last value N times along the T dimension
    last_value = r[:, -1:, :]
    last_value_repeated = last_value.repeat(1, N, 1)
    r = torch.cat((r, last_value_repeated), dim=1)

    # Repeat the last value N times along the T dimension
    last_value = y_d[:, -1:, :]
    last_value_repeated = last_value.repeat(1, N, 1)
    y_d = torch.cat((y_d, last_value_repeated), dim=1)

    print(y_d.shape)

    with open('evaporation_process_test_set_5%.pickle', 'rb') as handle:
        test_set = pickle.load(handle)
    with open('evaporation_process_test_set_5%_identified.pickle', 'rb') as handle:
        test_set_identified = pickle.load(handle)

    N = len(test_set)

    U = torch.zeros(N, 2, T, device=device)
    X = torch.zeros(N, 2, T + 1, device=device)
    Y = torch.zeros(N, 1, T, device=device)

    for idx in range(N-18):

        print('Test number:', idx)

        data = ([param.to(device) for param in test_set[idx]])
        data_identified = ([param.to(device) for param in test_set_identified[idx]])

        # Create an instance of the NMPC controller (nominal params)
        nmpc = NMPC(2, 2, N, data_identified)

        # Main loop for trajectory iteration
        X[idx,:,0] = y_d[0,0,:]

        for k in range(T):
            print(f"Step {k}, Current State: {X[idx,:,k]}")

            # Call the MPC block to get the optimal control input
            u_optimal = nmpc.optimize_control_sequence(X[idx,:,k], y_d[0, k:k+N+1])

            # Apply the optimal control to the real system dynamics
            X_dot = dynamics(X[idx, :, k], u_optimal, *data)
            X[idx, :, k + 1] = X[idx, :, k] + 1 * X_dot + torch.randn_like(X[idx, :, k], device=device) * .5
            Y[idx, :, k] = X[idx, 0, k] + torch.randn_like(X[idx, 0, k], device=device) * .5

    # Plot the reference and the final trajectory
    plt.figure(figsize=(10, 6))

    plt.figure(figsize=(5, 2.5))
    plt.plot(t.cpu().numpy(), r[0, :, 0].cpu().detach().T, label="$r$", c='k', linewidth=1)
    plt.plot(t.cpu().numpy(), y_d[0, :, 0].T, label="$y_d$", c='tab:red', linestyle='--', zorder=3)
    plt.plot(t.cpu().numpy(), Y[:, :, 0].T, c='tab:blue', alpha=1, label="no_label")
    plt.legend(['$r_1$', '$y_d$', '$y$'], loc='upper left')
    # plt.xlim([0, 0.43])
    # plt.ylim([19, 26])
    # plt.tick_params('x', labelbottom=False)
    plt.xlabel('$t$ [s]')
    plt.ylabel('$X_2$ [%]')
    plt.tight_layout()
    # plt.savefig("evaporation_evaporation_nmpc_identified.pdf")
    plt.show()
    # # Plot each state dimension
    # plt.subplot(211)
    # plt.plot(y_d[0, :, 0].cpu().numpy())
    # plt.plot(trajectory[:, 0].cpu().numpy())
    # plt.xlabel('Time Step')
    # plt.ylabel('State Value')
    # # plt.legend()
    # plt.grid(True)
    #
    # # Plot each state dimension
    # plt.subplot(212)
    # plt.plot(y_d[0, :, 1].cpu().numpy())
    # plt.plot(trajectory[:, 1].cpu().numpy())
    # plt.xlabel('Time Step')
    # plt.ylabel('State Value')
    # # plt.legend()
    # plt.grid(True)
    #
    # plt.show()


if __name__ == "__main__":
    main()