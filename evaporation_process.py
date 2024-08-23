import torch
import numpy as np
import matplotlib.pyplot as plt

def problem_data(perturbation, device="cuda:0"):
    """ Problem data, numeric constants,... adapted to work on GPU with torch """
    perturbation = torch.tensor(perturbation/100, dtype=torch.float32, device=device)
    data = {}

    data['a'] = torch.tensor(0.5616, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    data['b'] = torch.tensor(0.3126, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    data['c'] = torch.tensor(48.43, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    data['d'] = torch.tensor(0.507, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    data['e'] = torch.tensor(55.0, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    data['f'] = torch.tensor(0.1538, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    data['g'] = torch.tensor(90.0, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    data['h'] = torch.tensor(0.16, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    data['M'] = torch.tensor(20.0, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    data['C'] = torch.tensor(4.0, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    data['UA2'] = torch.tensor(6.84, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    data['Cp'] = torch.tensor(0.07, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    data['lam'] = torch.tensor(38.5, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    data['lams'] = torch.tensor(36.6, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    data['F1'] = torch.tensor(10.0, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    data['X1'] = torch.tensor(5.0, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    data['F3'] = torch.tensor(50.0, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    data['T1'] = torch.tensor(40.0, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))
    data['T200'] = torch.tensor(25.0, dtype=torch.float32, device=device) * (
            1.0 + perturbation * (torch.rand(1, device=device) * 2 - 1.0))

    return data




def intermediate_vars(x, u, data):
    """ Intermediate model variables (PyTorch version for GPU) """
    T2 = data['a'] * x[1] + data['b'] * x[0] + data['c']
    T3 = data['d'] * x[1] + data['e']
    T100 = data['f'] * u[0] + data['g']  # added noise
    UA1 = data['h'] * (data['F1'] + data['F3'])
    Q100 = UA1 * (T100 - T2)
    F100 = Q100 / data['lams']
    Q200 = data['UA2'] * (T3 - data['T200']) / (1.0 + data['UA2'] / (2.0 * data['Cp'] * u[1]))
    F5 = Q200 / data['lam']
    F4 = (Q100 - data['F1'] * data['Cp'] * (T2 - data['T1'])) / data['lam']
    F2 = data['F1'] - F4

    return {
        #'T2': T2,
        #'T3': T3,
        #'T100': T100,
        #'UA1': UA1,
        #'Q100': Q100,
        #'F100': F100,
        #'Q200': Q200,
        # 'Cp':data['Cp'],
        # 'T1':data['T1'],
        'F1': data['F1'],
        'F2': F2,
        'F4': F4,
        'F5': F5,
        'X1':data['X1'],
        'M':data['M'],
        'C':data['C']
    }



def dynamics(x, u, intermediate_data):
    """ System dynamics function (discrete time, PyTorch version for GPU) """
    # Cp = intermediate_data['Cp']
    # T1 = intermediate_data['T1']
    F1 = intermediate_data['F1']
    F2 = intermediate_data['F2']
    F4 = intermediate_data['F4']
    F5 = intermediate_data['F5']
    X1 = intermediate_data['X1']
    M = intermediate_data['M']
    C = intermediate_data['C']

    # State derivative calculations
    X2_dot = (F1 * X1 - F2 * x[0]) / M
    P2_dot = (F4 - F5) / C

    # Return state derivatives as a tensor
    xdot = torch.tensor([X2_dot, P2_dot], device=x.device)
    return xdot


def vars(device):
    """ System states and controls (PyTorch version for GPU) """
    x = torch.tensor([25.0, 49.743], dtype=torch.float32, device=device)  # Initial state, modify as needed
    u = torch.tensor([0.0, 0.0], dtype=torch.float32, device=device)  # Initial control input, modify as needed
    return x, u


def simulate_evaporation_process(u, data_perturbation,initial_state_perturbation, device="cuda:0", save_params=False, noisy = False):
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
    s_noisy = s.clone()  # Ensure to use clone


    if noisy :
        for i in range(num_steps):
            a = u[i, :]  # action
            # Store the nominal value
            x_n[i, :] = s
            # Calculate intermediate variables
            intermediate_data = intermediate_vars(s, a, data)
            # Dynamics equation
            x_dot = dynamics(s, a, intermediate_data)
            # Integrate dynamics using forward Euler integration
            s_next = s + Ts * x_dot
            # Store the noisy state and measure with additional noise
            x[i, :] = s_noisy
            y[i, :] = s_noisy + (torch.tensor([2.0, 2.0], device=device) * torch.randn(2, device=device))  # Measurement noise
            # Calculate intermediate variables for the noisy state
            intermediate_data = intermediate_vars(s_noisy, a, data)
            # Dynamics equation with noise
            x_dot = dynamics(s_noisy, a, intermediate_data)
            s_next_noisy = s_noisy + Ts * x_dot + (torch.tensor([0.5, 0.5], device=device) * torch.randn(2, device=device))
            # Update state
            s = s_next
            s_noisy = s_next_noisy
    else :
        for i in range(num_steps):
            a = u[i, :]  # action
            x_n[i, :] = s
            x[i,:]=s
            # Calculate intermediate variables
            intermediate_data = intermediate_vars(s, a, data)
            # Dynamics equation
            x_dot = dynamics(s, a, intermediate_data)
            # Integrate dynamics using forward Euler integration
            y[i, :] = s
            s_next = s + Ts * x_dot
            s = s_next



    if save_params:
        return x_n, x, y, intermediate_data
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
    data_pert_perc = 20
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
        #u_forced = constant_u + torch.zeros_like(prbs_signal, dtype=torch.float32, device="cuda:0")
        u_forced = torch.zeros_like(prbs_signal, dtype=torch.float32,  device="cuda:0")

        x_n, x, y = simulate_evaporation_process(u_forced, data_pert, state_pert, device="cuda:0", noisy = False)

        # Store the results in the preallocated arrays
        x_n_all[sim_id] = x_n.cpu().numpy()
        x_all[sim_id] = x.cpu().numpy()
        y_all[sim_id] = y.cpu().numpy()


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

