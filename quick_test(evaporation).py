import torch
from evaporation_process import problem_data, intermediate_vars, dynamics, vars
if __name__ == "__main__":

    device = "cuda:0"
    x_i = torch.zeros((2), device=device, dtype=torch.float32)
    u = torch.zeros((2), device=device, dtype=torch.float32)
    # Perturbation factor for initial conditions
    perturbation = 0

    # Simulate the system trajectory using the model
    data = problem_data(perturbation)
    print(data)
    intermediate_data = intermediate_vars(x_i, u, data)
    x_dot = dynamics(x_i, u, intermediate_data).reshape(-1)
    print(x_dot)
