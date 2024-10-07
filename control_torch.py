import numpy as np
import math
import torch
import random
from torch import zeros, complex, cos, sin, solve

class DiscreteTransferFunction(torch.nn.Module):
    def __init__(self, b, a, dt=1.0):
        super(DiscreteTransferFunction, self).__init__()
        self.b = torch.tensor(b, dtype=torch.float32)
        self.a = torch.tensor(a, dtype=torch.float32)
        self.dt = dt

    def forward(self, r):
        # Ensure the input is a tensor and has the correct dtype
        if not isinstance(r, torch.Tensor):
            r = torch.tensor(r, dtype=torch.float32)
        else:
            r = r.type(torch.float32)

        y = torch.zeros_like(r)
        # Compute the output using the difference equation
        for t in range(len(r)):
            for i in range(len(self.b)):
                if t - i >= 0:
                    y[t] += self.b[i] * r[t - i]
            for j in range(1, len(self.a)):
                if t - j >= 0:
                    y[t] -= self.a[j] * y[t - j]

        y = torch.cat((torch.tensor([0]), y[:-1]))

        return y * self.dt


def drss_matrices(
        states, inputs, outputs, strictly_proper=False, mag_range=(0.5, 0.97), phase_range=(0, math.pi / 2),
        dtype=torch.float32, device=None):
    """Generate a random state space using PyTorch, running on the GPU.

    This does the actual random state space generation expected from rss and
    drss.  cdtype is 'c' for continuous systems and 'd' for discrete systems.

    """

    # Probability of repeating a previous root.
    pRepeat = 0.05
    # Probability of choosing a real root.  Note that when choosing a complex
    # root, the conjugate gets chosen as well.  So the expected proportion of
    # real roots is pReal / (pReal + 2 * (1 - pReal)).
    pReal = 0.5
    # Probability that an element in B or C will not be masked out.
    pBCmask = 0.8
    # Probability that an element in D will not be masked out.
    pDmask = 0.3
    # Probability that D = 0.
    pDzero = 0.5

    # Check for valid input arguments.
    if states < 1 or states % 1:
        raise ValueError("states must be a positive integer.  states = %g." % states)
    if inputs < 1 or inputs % 1:
        raise ValueError("inputs must be a positive integer.  inputs = %g." % inputs)
    if outputs < 1 or outputs % 1:
        raise ValueError("outputs must be a positive integer.  outputs = %g." % outputs)

    # Create uniform distributions for magnitude and phase ranges
    mag_dist = torch.distributions.Uniform(mag_range[0], mag_range[1])
    phase_dist = torch.distributions.Uniform(phase_range[0], phase_range[1])

    # Make some poles for A. Preallocate a complex array.
    poles = torch.zeros(states, dtype=torch.complex128, device=device)
    i = 0

    while i < states:
        if torch.rand(1, device=device).item() < pRepeat and i != 0 and i != states - 1:
            # Small chance of copying poles, if we're not at the first or last element.
            if poles[i - 1].imag == 0:
                # Copy previous real pole.
                poles[i] = poles[i - 1]
                i += 1
            else:
                # Copy previous complex conjugate pair of poles.
                poles[i:i + 2] = poles[i - 2:i]
                i += 2
        elif torch.rand(1, device=device).item() < pReal or i == states - 1:
            # No-oscillation pole.
            poles[i] = mag_dist.sample((1,)).item()
            i += 1
        else:
            mag = mag_dist.sample((1,))
            phase = phase_dist.sample((1,))

            poles[i] = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
            poles[i + 1] = torch.complex(poles[i].real, -poles[i].imag)
            i += 2

    # Now put the poles in A as real blocks on the diagonal.
    A = torch.zeros((states, states), dtype=dtype, device=device)
    i = 0
    while i < states:
        if poles[i].imag == 0:
            A[i, i] = poles[i].real
            i += 1
        else:
            A[i, i] = A[i + 1, i + 1] = poles[i].real
            A[i, i + 1] = poles[i].imag
            A[i + 1, i] = -poles[i].imag
            i += 2

    # Finally, apply a transformation so that A is not block-diagonal.
    while True:
        T = torch.normal(0, 1, size=(states, states), device=device, dtype=A.dtype)
        try:
            A = torch.linalg.solve(T, A) @ T  # A = T \ A @ T
            break
        except RuntimeError as e:
            print(e)
            # In the unlikely event that T is rank-deficient, iterate again.
            pass

    # Make the remaining matrices.
    B = torch.normal(0, 1, size=(states, inputs), device=device, dtype=dtype)
    C = torch.normal(0, 1, size=(outputs, states), device=device, dtype=dtype)
    D = torch.normal(0, 1, size=(outputs, inputs), device=device, dtype=dtype)

    # Make masks to zero out some of the elements.
    while True:
        Bmask = torch.rand((states, inputs), device=device, dtype=dtype) < pBCmask
        if Bmask.any():  # Retry if we get all zeros.
            break
    while True:
        Cmask = torch.rand((outputs, states), device=device, dtype=dtype) < pBCmask
        if Cmask.any():  # Retry if we get all zeros.
            break
    if torch.rand(1, device=device, dtype=dtype).item() < pDzero:
        Dmask = torch.zeros((outputs, inputs), device=device, dtype=dtype)
    else:
        Dmask = torch.rand((outputs, inputs), device=device, dtype=dtype) < pDmask

    # Apply masks.
    B = B * Bmask
    C = C * Cmask
    D = D * Dmask if not strictly_proper else torch.zeros(D.shape, dtype=dtype, device=device)

    return A, B, C, D

def drss(nx, nu, ny, strictly_proper=True, device="cuda:0", dtype=torch.float32):
    """
    Generate random state-space matrices for a discrete-time linear system.
    Args:
        nx: Number of states
        nu: Number of inputs
        ny: Number of outputs
        device: Device to store tensors
        dtype: Data type of tensors
    Returns:
        A, B, C, D: State-space matrices
    """
    A = torch.randn(nx, nx, device=device, dtype=dtype) * 0.1
    B = torch.randn(nx, nu, device=device, dtype=dtype)
    C = torch.randn(ny, nx, device=device, dtype=dtype)
    D = torch.randn(ny, nu, device=device, dtype=dtype)

    if strictly_proper:
        D = D * 0.0

    # Ensure A is stable
    L_complex = torch.linalg.eigvals(A)
    max_eigval = torch.max(torch.abs(L_complex))
    if max_eigval >= 1:
        A = A / (max_eigval + 1.1)

    return A, B, C, D

def forced_response(A, B, C, D, u, x0=None, return_x=False):
    """
    Simulate the forced response of a discrete-time linear system.
    Args:
        A, B, C, D: State-space matrices
        u: Input sequence (T, nu)
        x0: Initial state (nx,)
    Returns:
        y: Output sequence (T, ny)
    """
    T, nu = u.shape
    nx = A.shape[0]
    ny = C.shape[0]

    # Convert x0 to tensor if it's not None
    if x0 is None:
        x0 = torch.zeros(nx, device=u.device, dtype=u.dtype)
    else:
        # x0 = torch.tensor(x0, device=u.device, dtype=u.dtype)
        if x0.shape[0] != nx:
            raise ValueError(f"Initial state x0 must have {nx} elements, but got {x0.shape[0]}")

    x = x0
    y = torch.zeros(T, ny, device=u.device, dtype=u.dtype)  # Preallocate tensor for outputs

    for t in range(0, T):  # Start from the second sample to avoid duplicating the initial output
        y[t] = C @ x + D @ u[t]
        x = A @ x + B @ u[t]

    if return_x:
        return y, x
    else:
        return y

def normalize(num, den, device='cuda'):
    """Normalize numerator/denominator of a continuous-time transfer function on GPU.

    Parameters
    ----------
    num: torch.Tensor
        Numerator of the transfer function. Can be a 2-D tensor to normalize
        multiple transfer functions.
    den: torch.Tensor
        Denominator of the transfer function. At most 1-D tensor.

    Returns
    -------
    num: torch.Tensor
        The numerator of the normalized transfer function. At least a 1-D
        tensor. A 2-D tensor if the input `num` is a 2-D tensor.
    den: torch.Tensor
        The denominator of the normalized transfer function.

    Notes
    -----
    Coefficients for both the numerator and denominator should be specified in
    descending exponent order (e.g., ``s^2 + 3s + 5`` would be represented as
    ``[1, 3, 5]``).
    """
    num = num.to(device)
    den = den.to(device)

    den = torch.atleast_1d(den)
    num = torch.atleast_2d(num)

    if den.ndimension() != 1:
        raise ValueError("Denominator polynomial must be rank-1 tensor.")
    if num.ndimension() > 2:
        raise ValueError("Numerator polynomial must be rank-1 or rank-2 tensor.")
    if torch.all(den == 0):
        raise ValueError("Denominator must have at least one nonzero element.")

    # Trim leading zeros in denominator, leave at least one
    den = torch.cat([den[torch.nonzero(den, as_tuple=True)[0][0]:],
                     torch.zeros(max(0, len(den) - len(den[torch.nonzero(den, as_tuple=True)[0][0]:])), device=device)])

    # Normalize transfer function
    den_0 = den[0]
    num = num / den_0
    den = den / den_0

    # Count numerator columns that are all zero
    leading_zeros = 0
    for col in num.T:
        if torch.allclose(col, torch.tensor(0.0, device=device), atol=1e-14):
            leading_zeros += 1
        else:
            break

    # Trim leading zeros of numerator
    if leading_zeros > 0:
        # Make sure at least one column remains
        if leading_zeros == num.shape[1]:
            leading_zeros -= 1
        num = num[:, leading_zeros:]

    # Squeeze first dimension if singular
    if num.shape[0] == 1:
        num = num.squeeze(0)

    return num, den


def tf2ss(num, den, device='cuda'):
    """
    Convert a transfer function to state-space representation using canonical form.

    Args:
        num (torch.Tensor): Numerator coefficients.
        den (torch.Tensor): Denominator coefficients.
        device (str): Device to run on ('cuda' or 'cpu').

    Returns:
        Tuple: State-space matrices (A, B, C, D).
    """
    device = torch.device(device)

    num, den = normalize(num, den)  # Normalize the input

    num = num.reshape(-1, 1)
    den = den.reshape(-1, 1)

    M = num.shape[0]
    K = den.shape[0]

    if M > K:
        raise ValueError("Improper transfer function. `num` is longer than `den`.")

    if M == 0 or K == 0:  # Null system
        return (torch.zeros((0, 0), dtype=torch.float32, device=device),
                torch.zeros((0, 0), dtype=torch.float32, device=device),
                torch.zeros((0, 0), dtype=torch.float32, device=device),
                torch.zeros((0, 0), dtype=torch.float32, device=device))

    # Pad numerator to have same number of columns as denominator
    num = torch.cat((torch.zeros((num.shape[0], K - M), dtype=num.dtype, device=device), num), dim=1)

    if num.shape[-1] > 0:
        D = num[0].unsqueeze(1)  # Create 2D tensor for D
    else:
        D = torch.tensor([[0]], dtype=torch.float32, device=device)

    if K == 1:
        D = D.reshape(num.shape)
        return (torch.zeros((1, 1), dtype=torch.float32, device=device),
                torch.zeros((1, D.shape[1]), dtype=torch.float32, device=device),
                torch.zeros((D.shape[0], 1), dtype=torch.float32, device=device),
                D)

    # Create A matrix
    A = torch.zeros((K - 1, K - 1), dtype=torch.float32, device=device)
    A[0, :] = -den[1:] / den[0]
    A[1:, :-1] = torch.eye(K - 2, dtype=torch.float32, device=device)

    # Create B matrix
    B = torch.eye(K - 1, 1, dtype=torch.float32, device=device)

    # Create C matrix
    C = num[1:] - torch.outer(num[0].reshape(-1), den[1:].reshape(-1))

    # Ensure D is in the correct shape
    D = D.reshape(C.shape[0], B.shape[1])

    return A, B, C, D

def c2d(A, B, C, D, dt, device='cuda'):
    """
    Convert continuous-time state-space matrices to discrete-time using the bilinear transform.

    Args:
        A (torch.Tensor): Continuous-time state matrix of shape (n, n).
        B (torch.Tensor): Continuous-time input matrix of shape (n, m).
        C (torch.Tensor): Continuous-time output matrix of shape (p, n).
        D (torch.Tensor): Continuous-time feedthrough matrix of shape (p, m).
        T (float): Sampling period.
        device (str): Device to run on ('cuda' or 'cpu').

    Returns:
        A_d (torch.Tensor): Discrete-time state matrix.
        B_d (torch.Tensor): Discrete-time input matrix.
        C_d (torch.Tensor): Discrete-time output matrix.
        D_d (torch.Tensor): Discrete-time feedthrough matrix.
    """
    # Ensure matrices are on the correct device
    A = A.to(device)
    B = B.to(device)
    C = C.to(device)
    D = D.to(device)

    alpha = 0.5

    # Compute I - alpha * dt * a
    I = torch.eye(A.size(0), device=device)
    ima = I - alpha * dt * A

    # Compute ad and bd by solving linear systems
    I_alpha = I + (1.0 - alpha) * dt * A
    A_d = torch.linalg.solve(ima, I_alpha)
    B_d = torch.linalg.solve(ima, dt * B)

    # Compute cd and dd
    C_d = torch.linalg.solve(ima.T, C.T).T
    D_d = D + alpha * torch.matmul(C, B_d)

    return A_d, B_d, C_d, D_d

def perturb_matrices(A, B, C, D, percentage, device='cuda'):
    """
    Perturb the values of A, B, C, D matrices by a fixed percentage.

    Args:
        A, B, C, D: State-space matrices (torch.Tensor).
        percentage: The percentage by which to perturb the matrices (float).
        device: The device to perform the perturbation on (str).

    Returns:
        A_perturbed, B_perturbed, C_perturbed, D_perturbed: Perturbed matrices.
    """
    # Ensure percentage is a fraction
    percentage /= 100.0

    # # Generate random perturbations
    # perturb_A = torch.randn_like(A, device=device) * percentage * A
    # perturb_B = torch.randn_like(B, device=device) * percentage * B
    # perturb_C = torch.randn_like(C, device=device) * percentage * C
    # perturb_D = torch.randn_like(D, device=device) * percentage * D
    #
    # # Apply perturbations
    # A_perturbed = A + perturb_A
    # B_perturbed = B + perturb_B
    # C_perturbed = C + perturb_C
    # D_perturbed = D + perturb_D
    #
    # # Clip the perturbations of A between 0 and 1
    # A_perturbed = torch.clamp(A_perturbed, min=0, max=1-1e-3)
    #
    # return A_perturbed, B_perturbed, C_perturbed, D_perturbed

    # Move matrices to the desired device
    A, B, C, D = A.to(device), B.to(device), C.to(device), D.to(device)

    # Compute eigenvalues and eigenvectors of A
    eigvals, eigvecs = torch.linalg.eig(A)

    # Convert to real values if there are no complex numbers
    eigvals_real = eigvals.real
    eigvals_imag = eigvals.imag

    # Perturb eigenvalues
    perturb_real = torch.randn_like(eigvals_real, device=device) * percentage * eigvals_real
    perturb_imag = torch.randn_like(eigvals_imag, device=device) * percentage * eigvals_imag

    eigvals_real_perturbed = eigvals_real + perturb_real
    eigvals_imag_perturbed = eigvals_imag + perturb_imag

    # Combine perturbed eigenvalues into a complex tensor
    eigvals_perturbed = torch.complex(eigvals_real_perturbed, eigvals_imag_perturbed)

    # Ensure eigenvalues remain inside the unit circle
    for i in range(len(eigvals_perturbed)):
        if eigvals_perturbed[i].abs() >= 1:
            eigvals_perturbed[i] = eigvals_perturbed[i] / (eigvals_perturbed[i].abs() + 1e-3)

    # Reconstruct A_perturbed from the perturbed eigenvalues and original eigenvectors
    A_perturbed = eigvecs @ torch.diag(eigvals_perturbed) @ torch.linalg.inv(eigvecs)

    # Perturb B, C, D matrices normally
    perturb_B = torch.randn_like(B, device=device) * percentage * B
    perturb_C = torch.randn_like(C, device=device) * percentage * C
    perturb_D = torch.randn_like(D, device=device) * percentage * D

    B_perturbed = B + perturb_B
    C_perturbed = C + perturb_C
    D_perturbed = D + perturb_D

    return A_perturbed.real, B_perturbed, C_perturbed, D_perturbed

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def perturb_parameters(param, percentage, device = 'cuda'):
    set_seed(random.randint(1, 500))
    perturb = (2*torch.rand_like(param, device=device)-1) * (percentage / 100.0) * param
    return param + perturb