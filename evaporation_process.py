import torch
import numpy as np
import matplotlib.pyplot as plt


def problem_data(perturbation, device="cuda:0"):
    """ Problem data, numeric constants,... adapted to work on GPU with torch """

    perturbation = torch.tensor(perturbation / 100, dtype=torch.float32, device=device)

    # Define all variables directly
    a = torch.tensor(0.5616, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    b = torch.tensor(0.3126, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    c = torch.tensor(48.43, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    d = torch.tensor(0.507, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    e = torch.tensor(55.0, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    f = torch.tensor(0.1538, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    g = torch.tensor(90.0, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    h = torch.tensor(0.16, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    M = torch.tensor(20.0, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    C = torch.tensor(4.0, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    UA2 = torch.tensor(6.84, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    Cp = torch.tensor(0.07, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    lam = torch.tensor(38.5, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    lams = torch.tensor(36.6, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    F1 = torch.tensor(10.0, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    X1 = torch.tensor(5.0, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    F3 = torch.tensor(50.0, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    T1 = torch.tensor(40.0, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    T200 = torch.tensor(25.0, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))

    # Return all values individually
    return a, b, c, d, e, f, g, h, M, C, UA2, Cp, lam, lams, F1, X1, F3, T1, T200


# def intermediate_vars(x, u, a, b, c, d, e, f, g, h, M, C, UA2, Cp, lam, lams, F1, X1, F3, T1, T200):
#     """ Intermediate model variables (PyTorch version for GPU) """
#
#     # Calculate intermediate variables using the provided parameters
#     T2 = a * x[1] + b * x[0] + c
#     T3 = d * x[1] + e
#     T100 = f * u[0] + g  # added noise
#     UA1 = h * (F1 + F3)
#     Q100 = UA1 * (T100 - T2)
#     F100 = Q100 / lams
#     Q200 = UA2 * (T3 - T200) / (1.0 + UA2 / (2.0 * Cp * u[1]))
#     F5 = Q200 / lam
#     F4 = (Q100 - F1 * Cp * (T2 - T1)) / lam
#     F2 = F1 - F4
#
#     # Return all relevant variables individually
#     return T2, T3, T100, UA1, Q100, F100, Q200, F1, F2, F4, F5, X1, M, C


def dynamics(x, u, a, b, c, d, e, f, g, h, M, C, UA2, Cp, lam, lams, F1, X1, F3, T1, T200):
    """ System dynamics function (discrete time, PyTorch version for GPU) """

    # Calculate intermediate variables using the provided parameters
    T2 = a * x[1] + b * x[0] + c
    T3 = d * x[1] + e
    T100 = f * u[0] + g  # added noise
    UA1 = h * (F1 + F3)
    Q100 = UA1 * (T100 - T2)
    F100 = Q100 / lams
    Q200 = UA2 * (T3 - T200) / (1.0 + UA2 / (2.0 * Cp * u[1]))
    F5 = Q200 / lam
    F4 = (Q100 - F1 * Cp * (T2 - T1)) / lam
    F2 = F1 - F4

    # State derivative calculations
    X2_dot = (F1 * X1 - F2 * x[0].clone()) / M
    P2_dot = (F4 - F5) / C

    # # Return state derivatives as a tensor
    xdot = torch.cat((X2_dot, P2_dot), dim=0).to(x.device)
    return xdot


def vars(device):
    """ System states and controls (PyTorch version for GPU) """
    x = torch.tensor([25.0, 49.743], dtype=torch.float32, device=device)  # Initial state, modify as needed
    u = torch.tensor([0.0, 0.0], dtype=torch.float32, device=device)  # Initial control input, modify as needed
    return x, u


def simulate_evaporation_process(u, data_perturbation, initial_state_perturbation, device="cuda:0", save_params=False, noisy = False):
    Ts = 1  # Time step
    data = problem_data(data_perturbation, device)

    x_0, u_0 = vars(device)

    # Initialize tensors to store the trajectory
    num_steps = u.size(0)
    x_n = torch.zeros((num_steps, 2), device=device)
    x = torch.zeros((num_steps, 2), device=device)
    y = torch.zeros((num_steps, 2), device=device)

    # Perturb the initial state
    x_0 = x_0 * (1. + (initial_state_perturbation/100) * (torch.rand(2, device=device) * 2 - 1.))
    s = x_0

    for i in range(num_steps):

        print('time instant:', i)
        a = u[i, :]  # action
        x_n[i, :] = s
        x[i,:]=s
        # Calculate intermediate variables
        # intermediate_data = intermediate_vars(s, a, *data)
        # Dynamics equation
        x_dot = dynamics(s, a, *data)

        # Integrate dynamics using forward Euler integration
        y[i, :] = s
        s_next = s + Ts * x_dot
        s = s_next

    if save_params:
        return x_n, x, y, data
    else:
        return x_n, x, y

def generate_prbs(num_steps, perturbation_range=20):
    """Generate a Perturbed Random Binary Signal (PRBS)"""
    prbs = torch.randint(0, 2, (num_steps, 2), dtype=torch.float32,  device="cuda:0") * 2 - 1  # Randomly choose -1 or 1
    prbs *= perturbation_range  # Scale to ±20
    return prbs


import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Set up parameters
    num_steps = 500  # Number of time steps per simulation
    num_simulations = 10 # Number of simulations to run

    # Generate inputs for simulation
    constant_u = torch.tensor([191.713, 215.888], device="cuda:0")

    # Perturbation factor for initial conditions
    data_pert_perc = 0
    state_pert = 20

    # Initialize arrays to store results of each simulation
    x_n_all = np.zeros((num_simulations, num_steps, 2))
    x_all = np.zeros((num_simulations, num_steps, 2))
    y_all = np.zeros((num_simulations, num_steps, 2))

    for sim_id in range(num_simulations):
        # Simulate the system trajectory using the model
        prbs_signal = generate_prbs(num_steps)

        # use constant input u or perturbed with PRBS
        #u_forced = constant_u + prbs_signal  # Perturb the steady-state input
        u_forced = constant_u + torch.zeros_like(prbs_signal, dtype=torch.float32, device="cuda:0")
        # u_forced = torch.zeros_like(prbs_signal, dtype=torch.float32,  device="cuda:0")

        x_n, x, y = simulate_evaporation_process(u_forced, data_pert_perc, state_pert, device="cuda:0", noisy = False)

        # Store the results in the preallocated arrays
        x_n_all[sim_id] = x_n.detach().cpu().numpy()
        x_all[sim_id] = x.detach().cpu().numpy()
        y_all[sim_id] = y.detach().cpu().numpy()


    # Plot the results for all simulations
    plt.figure(figsize=(10, 5))
    # Plot the first measurement for all simulations
    plt.subplot(211)
    plt.plot(y_all[:, :, 0].T, color='blue', alpha=0.5)  # Plotting the first measurement across all simulations
    plt.xlabel(r'$t \ [s]$')
    #plt.ylabel(r'$x_1^0$')
    plt.ylabel(r'$y_1$')
    plt.ylim([-10,40])
    plt.grid()
    # Plot the second measurement for all simulations
    plt.subplot(212)
    plt.plot(y_all[:, :, 1].T, color='blue', alpha=0.5)  # Plotting the second measurement across all simulations
    plt.xlabel(r'$t \ [s]$')
    #plt.ylabel(r'$x_2^0$')
    plt.ylabel(r'$y_2$')
    plt.ylim([30, 120])
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
