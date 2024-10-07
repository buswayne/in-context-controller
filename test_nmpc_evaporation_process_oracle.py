#%%
from torch.utils.data import DataLoader
from dataset_CL_evaporation_process import EvaporationDataset
from evaporation_process import problem_data, dynamics

import torch
import matplotlib.pyplot as plt
import pickle
from nmpc_evaporation_process import NMPC
import wandb

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="in-context controller",
    name="nmpc-oracle"
)

torch.cuda.set_device('cuda:3')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define state and control dimensions
nx = 2
nu = 2
# Define simulation parameters
T = 100
t = torch.arange(T)
N = 5  # Prediction horizon

# Initialize the initial state, reference state, and system parameters
x0 = torch.tensor([25.0, 49.0], dtype=torch.float32, device=device)
x_ref = torch.tensor([30.0, 40.0], dtype=torch.float32, device=device)

# Initialize the model
dataset = EvaporationDataset(seq_len=T, ts=1, seed=42, data_perturb_percentage=0, device=device)
dataloader = DataLoader(dataset, batch_size=1)

with open('evaporation_process_test_set_5%2525.pickle', 'rb') as handle:
    test_set = pickle.load(handle)
with open('evaporation_process_r.pickle', 'rb') as handle:
    batch_r = pickle.load(handle)
with open('evaporation_process_y_d.pickle', 'rb') as handle:
    batch_y_d = pickle.load(handle)
# with open('evaporation_process_test_set_5%_identified.pickle', 'rb') as handle:
#     test_set_identified = pickle.load(handle)

M = len(test_set)

U = torch.zeros(M, 2, T, device=device)
X = torch.zeros(M, 2, T + 1, device=device)
Y = torch.zeros(M, 1, T, device=device)

for idx in range(M):

    print('Test number:', idx)

    # _, r, y_d = next(iter(dataloader))
    r = batch_r[idx:idx+1, :, :]
    y_d = batch_y_d[idx:idx+1, :, :]

    # Repeat the last value N times along the T dimension
    last_value = r[:, -1:, :]
    last_value_repeated = last_value.repeat(1, N, 1)
    r = torch.cat((r, last_value_repeated), dim=1)

    # Repeat the last value N times along the T dimension
    last_value = y_d[:, -1:, :]
    last_value_repeated = last_value.repeat(1, N, 1)
    y_d = torch.cat((y_d, last_value_repeated), dim=1)

    data = ([param.to(device) for param in test_set[idx]])
    # data_identified = ([param.to(device) for param in test_set_identified[idx]])

    # Create an instance of the NMPC controller (nominal params)
    nmpc = NMPC(2, 2, N, data)

    # Main loop for trajectory iteration
    X[idx,:,0] = y_d[0,0,:]

    for k in range(T):
        print(f"Step {k}, Current State: {X[idx,:,k]}")

        # Call the MPC block to get the optimal control input
        u_optimal = nmpc.optimize_control_sequence(X[idx,:,k], y_d[0, k:k+N+1])

        U[idx, :, k] = u_optimal
        # Apply the optimal control to the real system dynamics
        X_dot = dynamics(X[idx, :, k], u_optimal, *data)
        X[idx, :, k + 1] = X[idx, :, k] + 1 * X_dot + torch.randn_like(X[idx, :, k], device=device) * .1
        Y[idx, :, k] = X[idx, 0, k] + torch.randn_like(X[idx, 0, k], device=device) * .1

nmpc_oracle_results = {'X':X, 'U':U, 'Y':Y}

with open('nmpc_oracle_results_mixed', 'wb') as handle:
    pickle.dump(nmpc_oracle_results, handle)
# #%%
# # Plot the reference and the final trajectory
# plt.figure(figsize=(10, 6))
#
# plt.figure(figsize=(5, 2.5))
# plt.plot(t.cpu().detach().numpy(), r[:, :-N, 0].cpu().detach().T, label="$r$", c='k', linewidth=1)
# plt.plot(t.cpu().detach().numpy(), y_d[:, :-N, 0].cpu().detach().T, label="$y_d$", c='tab:red', linestyle='--', zorder=3)
# plt.plot(t.cpu().detach().numpy(), Y[:, 0, :].T.cpu().detach(), c='tab:blue', alpha=1, label="no_label")
# plt.legend(['$r_1$', '$y_d$', '$y$'], loc='upper left')
# # plt.xlim([0, 0.43])
# plt.ylim([19, 26])
# # plt.tick_params('x', labelbottom=False)
# plt.xlabel('$t$ [s]')
# plt.ylabel('$X_2$ [%]')
# plt.tight_layout()
# # plt.savefig("evaporation_evaporation_nmpc_identified.pdf")
# plt.show()
# # # Plot each state dimension
# # plt.subplot(211)
# # plt.plot(y_d[0, :, 0].cpu().numpy())
# # plt.plot(trajectory[:, 0].cpu().numpy())
# # plt.xlabel('Time Step')
# # plt.ylabel('State Value')
# # # plt.legend()
# # plt.grid(True)
# #
# # # Plot each state dimension
# # plt.subplot(212)
# # plt.plot(y_d[0, :, 1].cpu().numpy())
# # plt.plot(trajectory[:, 1].cpu().numpy())
# # plt.xlabel('Time Step')
# # plt.ylabel('State Value')
# # # plt.legend()
# # plt.grid(True)
# #
# # plt.show()
#%%
