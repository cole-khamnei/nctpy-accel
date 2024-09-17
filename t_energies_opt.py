import torch
import numpy as np
import warnings

def get_cti_A_components(A_norm, T=1, rho=1, dt = 0.001, device="cpu", **extras):
    """ """
    devp = dict(device=torch.device(device))
    #TODO: add assertations statements to check dims

    A_norm = torch.tensor(A_norm.astype("float32"), **devp)
    n_nodes = A_norm.shape[-1]
    multiple_A = len(A_norm.shape) == 3

    I_b = torch.eye(n_nodes, **devp)
    if multiple_A:
        T_dims, add_dim = (1, 2), 1
        I_b = I_b.repeat(A_norm.shape[0], 1, 1)
    else:
        T_dims, add_dim = (0, 1), 0

    S_b, B_b, B_b_T = I_b, I_b, I_b.transpose(*T_dims)
    T, rho, dt = torch.tensor([T, rho, dt], **devp)

    M = torch.cat([torch.cat([A_norm, (-B_b @ B_b_T) / (2 * rho)], dim=1 + add_dim),
                   torch.cat([- 2 * S_b, -A_norm.transpose(*T_dims)], dim=1 + add_dim)],
                   dim=add_dim)

    E = torch.linalg.matrix_exp(M * T)
    Ad = torch.linalg.matrix_exp(M * dt)

    if multiple_A:
        E11, E12 = E[:, :n_nodes][:, :, :n_nodes], E[:, :n_nodes][:, :, n_nodes:]
    else:
        E11, E12 = E[:n_nodes][:, :n_nodes], E[:n_nodes][:, n_nodes:]

    dd_op = torch.cat([E11 - I_b, E12], dim=1 + add_dim)

    return M, E11, E12, Ad, S_b, B_b_T, dd_op


def get_cti_block(A_norms, x0s, xfs, T=1, rho=1, dt = 0.001, device="cpu", intermediates=None, **extras):
    """ """
    if intermediates is None:
        assert len(A_norms) == len(x0s) or (len(A_norms.shape) == 2)
        M, E11, E12, Ad, S, B_T, dd_op = get_cti_A_components(A_norms, T=T, rho=rho, dt=dt, device=device)
    else:
        M, E11, E12, Ad, S_b, B_b_T, dd_op = intermediates

    devp = dict(device=torch.device(device))
    x0s_b = torch.tensor(x0s.reshape(*x0s.shape, 1).astype("float32"), **devp)
    xfs_b = torch.tensor(xfs.reshape(*xfs.shape, 1).astype("float32"), **devp)
    xrs_b = torch.zeros(x0s_b.shape, **devp)

    n_batch, n_nodes = x0s.shape
    n_integrate_steps = int(np.round(T / dt)) + 1

    c = torch.cat([torch.zeros(xrs_b.shape, **devp), 2 * S @ xrs_b], dim=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = torch.linalg.solve(M, c)
        dd = xfs_b - (E11 @ x0s_b) - (dd_op @ c)
        l0 = torch.linalg.solve(E12, dd)

    # Make Big matrices
    big_I = torch.eye(2 * n_nodes, **devp)
    Bd = (Ad - big_I) @ c

    # Integrate trajectory
    z = torch.zeros((n_batch, 2 * n_nodes, n_integrate_steps), **devp)
    z[:, :, 0:1] = torch.cat([x0s_b, l0], dim=1)
    for i in range(1, n_integrate_steps):
        z[:, :, i:i+1] = Ad @ z[:, :, i-1:i] + Bd

    x = z[:, :n_nodes, :]
    u = (- B_T @ z[:, n_nodes:, :]) / (2 * rho)
    E = torch.sum(u ** 2, dim=(1, 2)).cpu().numpy()

    err_xf = torch.linalg.norm(x[:, :, -1:] - xfs_b, dim=1)
    err_costate = torch.linalg.norm(E12 @ l0 - dd, dim=1)
    err = torch.cat([err_xf, err_costate], dim=1).cpu().numpy()

    if ret_mem:
        return torch.cuda.mem_get_info()

    if device == "cuda":
        torch.cuda.empty_cache()
    return E, x.transpose(1, 2).cpu().numpy(), u.transpose(1, 2).cpu().numpy(), err



def get_max_batch_size(n_nodes, device):
    """ """
    if get_device_type(device) != "cuda":
        return 3

    mem_scale = get_GPU_free_memory(device) / 25_429_606_400
    size_scale = 400 * 400 / (n_nodes * n_nodes)

    return int(np.floor(450 * mem_scale * size_scale / 10) * 10)


def get_cti_batch(A_norms, x0s, xfs, T=1, dt=0.001, rho=1, device=None):
    """ """
    device = get_device(device)
    # Pass subject let arguments into get_cti_batch:
    n_states, n_nodes = x0s.shape
    n_integrate_steps = int(np.round(T / dt)) + 1
    n_batch = max(min(get_max_batch_size(n_nodes, device), 200), 1)
    n_blocks = int(np.ceil(n_states / n_batch))

    E_s = np.empty(n_states)
    x_s = np.empty((n_states, n_integrate_steps, n_nodes))
    u_s = np.empty((n_states, n_integrate_steps, n_nodes))
    err_s = np.empty((n_states, 2))

    multiple_A = len(A_norms.shape) == 3
    if not multiple_A:
        single_intermediates = get_cti_A_components(A_norms, T=T, rho=rho, dt=dt, device=device)

    for i in tqdm(range(n_blocks), leave=False, desc="Single trajectory set"):
        sl = slice(n_batch * i, min(n_batch * (i + 1), n_states))

        if multiple_A:
            intermediates = get_cti_A_components(A_norms[sl], T=T, rho=rho, dt=dt, device=device)
        else:
            intermediates = single_intermediates

        block_results = get_cti_block(None, x0s[sl], xfs[sl], intermediates=intermediates,
                                      T=T, dt=dt, rho=rho, device=device.type)
        E_s[sl], x_s[sl], u_s[sl], err_s[sl] = block_results

    return E_s, x_s, u_s, err_s
