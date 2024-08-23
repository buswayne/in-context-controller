import torch

def problem_data(perturbation, device="cuda:0"):
    """ Problem data, numeric constants,... adapted to work on GPU with torch """
    perturbation = torch.tensor(perturbation, dtype=torch.float32, device=device)
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
    xdot = torch.stack([X2_dot, P2_dot])

    return xdot


def vars(device):
    """ System states and controls (PyTorch version for GPU) """
    x = torch.tensor([25.0, 49.743], dtype=torch.float32, device=device)  # Initial state, modify as needed
    u = torch.tensor([0.0, 0.0], dtype=torch.float32, device=device)  # Initial control input, modify as needed
    return x, u