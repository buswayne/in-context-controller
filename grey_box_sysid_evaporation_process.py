import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_CL_evaporation_process import EvaporationDataset
from signals_torch import steps_sequence, prbs
from evaporation_process import dynamics
from torch_utils import select_gpu_with_most_free_memory
import matplotlib.pyplot as plt
import pickle
import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="in-context controller"
)

class GreyBoxModel(nn.Module):
    def __init__(self):
        super(GreyBoxModel, self).__init__()

        # Initialize the parameters for identification as nn.Parameters
        self.params = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5616, dtype=torch.float32, device='cuda')),
            nn.Parameter(torch.tensor(0.3126, dtype=torch.float32, device='cuda')),
            nn.Parameter(torch.tensor(48.43, dtype=torch.float32, device='cuda')),
            nn.Parameter(torch.tensor(0.507, dtype=torch.float32, device='cuda')),
            nn.Parameter(torch.tensor(55.0, dtype=torch.float32, device='cuda')),
            nn.Parameter(torch.tensor(0.1538, dtype=torch.float32, device='cuda')),
            nn.Parameter(torch.tensor(90.0, dtype=torch.float32, device='cuda')),
            nn.Parameter(torch.tensor(0.16, dtype=torch.float32, device='cuda')),
            nn.Parameter(torch.tensor(20.0, dtype=torch.float32, device='cuda')),
            nn.Parameter(torch.tensor(4.0, dtype=torch.float32, device='cuda')),
            nn.Parameter(torch.tensor(6.84, dtype=torch.float32, device='cuda')),
            nn.Parameter(torch.tensor(0.07, dtype=torch.float32, device='cuda')),
            nn.Parameter(torch.tensor(38.5, dtype=torch.float32, device='cuda')),
            nn.Parameter(torch.tensor(36.6, dtype=torch.float32, device='cuda')),
            nn.Parameter(torch.tensor(10.0, dtype=torch.float32, device='cuda')),
            nn.Parameter(torch.tensor(5.0, dtype=torch.float32, device='cuda')),
            nn.Parameter(torch.tensor(50.0, dtype=torch.float32, device='cuda')),
            nn.Parameter(torch.tensor(40.0, dtype=torch.float32, device='cuda')),
            nn.Parameter(torch.tensor(25.0, dtype=torch.float32, device='cuda'))
        ])

    # def forward(self, x, u):
    #     """
    #     Forward pass that predicts the system dynamics.
    #     """
    #     x_dot = dynamics(x, u, *self.params)
    #     selfx = x + 1 * x_dot
    #     y = x
    #
    #     return y

    def optimize_parameters(self, x_data, u_data, y_data, lr=1e-3, num_iter=100):
        """
        Optimize the parameters based on experimental input-output data.

        Args:
        - x_data (torch.Tensor): Initial state vectors (shape: (num_samples, state_dim))
        - u_data (torch.Tensor): Input vectors (shape: (num_samples, control_dim))
        - y_data (torch.Tensor): Output (target) vectors (shape: (num_samples, state_dim))
        - lr (float): Learning rate for the optimizer
        - num_iter (int): Number of optimization iterations

        Returns:
        - optimized_params (list): List of optimized parameters
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(num_iter):
            optimizer.zero_grad()

            x_pred = torch.zeros_like(x_data)
            y_pred = torch.zeros_like(y_data)

            x_pred = x_data[:, 0]

            # Predict the states using the dynamics function
            for k in range(u_data.size(1)):
                x_pred_dot = dynamics(x_pred, u_data[:, k], *self.params)
                x_pred = x_pred + 1 * x_pred_dot
                y_pred[:, k] = x_pred[0]

            # Compute the loss between predicted states and actual states
            mse_loss = nn.MSELoss()
            loss = mse_loss(y_pred, y_data)

            # Backpropagation
            loss.backward()

            # Update parameters
            optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')

        # Return the optimized parameters
        optimized_params = [param.detach() for param in self.params]
        return optimized_params


def main():

    torch.autograd.set_detect_anomaly(True)
    # Example usage
    best_gpu = select_gpu_with_most_free_memory()
    device = f'cuda:{best_gpu}' if best_gpu is not None else 'cpu'

    # dataset = EvaporationDataset(seq_len=100, ts=1, seed=42, data_perturb_percentage=5, device=device)
    # dataloader = DataLoader(dataset, batch_size=1)
    # data, _, _ = next(iter(dataloader))

    with open('evaporation_process_test_set_5%.pickle', 'rb') as handle:
        test_set = pickle.load(handle)

    T = 1000


    test_set_identified = {}

    for idx in range(len(test_set)):

        print('Test number:', idx)
        data = ([param.to(device) for param in test_set[idx]])
        # open loop experiment
        U = torch.zeros(2, T, device=device)
        X = torch.zeros(2, T+1, device=device)
        Y = torch.zeros(1, T, device=device)

        constant_u = torch.tensor([191.713, 215.888], device="cuda")
        U[0,:] = steps_sequence(T, 1, 150, 250, 1, 10).T
        U[1, :] = steps_sequence(T, 1, 150, 250, 1, 10).T
        # U[0, :] = constant_u[0] + prbs(T).T
        # U[1, :] = constant_u[1] + prbs(T).T
        X[:,0] = torch.tensor([25.0, 49.743]) #

        for k in range(T):
            X_dot = dynamics(X[:,k], U[:,k], *data)
            X[:,k+1] = X[:,k] + 1 * X_dot + torch.randn_like(X[:,k], device=device)*.5
            Y[:,k] = X[0,k] + torch.randn_like(X[0,k], device=device)*.5

        # Initialize and train the grey-box model
        greybox_model = GreyBoxModel().to(device)

        optimized_params = greybox_model.optimize_parameters(X, U, Y)

        test_set_identified[idx] = optimized_params

        print(data)
        print(optimized_params)
        #
        # X_hat = torch.zeros(2, T+1, device=device)
        # Y_hat = torch.zeros(1, T, device=device)
        # X_hat[:,0] = torch.tensor([25.0, 49.743])
        #
        # for k in range(T):
        #     X_dot = dynamics(X_hat[:,k], U[:,k], *optimized_params)
        #     X_hat[:,k+1] = X_hat[:,k] + 1 * X_dot
        #     Y_hat[:,k] = X_hat[0,k]
        #
        #
        #
        # plt.figure(figsize=(10,6))
        #
        # plt.subplot(211)
        # plt.plot(X[0,:].T.cpu().detach().numpy(), label='$x_1$')
        # plt.plot(X_hat[0, :].T.cpu().detach().numpy(), label='$\hat{x}_1$')
        # # plt.plot(X[1, :].T.cpu().detach().numpy(), label='$x_2$')
        # plt.legend()
        #
        # plt.subplot(212)
        # plt.plot(U[0,:].T.cpu().detach().numpy(), label='$u_1$')
        # plt.plot(U[1, :].T.cpu().detach().numpy(), label='$u_2$')
        # plt.legend()
        #
        # plt.show()

    with open('evaporation_process_test_set_5%_identified.pickle', 'wb') as handle:
        pickle.dump(test_set_identified, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()