import argparse
import os
import sys
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import warnings
import numpy as np
import dill as pickle

from time import time

import __main__ as main
if not hasattr(main, '__file__'):
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


VALID_SYSTEMS = ["discrete", "continuous"]

def get_device_type(device):
    """ """
    return device if isinstance(device, str) else device.type

def get_GPU_free_memory(device):
    """ """
    device_type = get_device_type(device)
    if device_type == "cuda":
        free_mem, total = torch.cuda.mem_get_info()
        return free_mem
    elif device_type == "mps":
        total = torch.mps.driver_allocated_memory()
        allocated = torch.mps.current_allocated_memory()
        return total - allocated
    else:
        raise NotImplementedError


def get_device(device=None):
    """ """
    if device:
        return torch.device(device)
    default = ("mps" if torch.backends.mps.is_available() else "cuda"
                     if torch.cuda.is_available() else "cpu")
    return torch.device(default)


def matrix_norm(A, c=1, system="continuous"):
    """ """
    assert system in VALID_SYSTEMS, f"Invalid system '{system}', valid sytems: {VALID_SYSTEMS}"

    # eigenvalue decomposition
    w, _ = np.linalg.eigh(A)
    l = np.abs(w).max()

    # Matrix normalization for discrete-time systems
    A_norm = A / (c + l)

    if system == 'continuous':
        # for continuous-time systems
        A_norm = A_norm - np.eye(A.shape[0])

    return A_norm


def get_cti_torch(A_norm, x0, xf, B=None, S=None, T=1, rho=1, dt = 0.001, device="cpu", **extras):
    """ """
    devp = dict(device=torch.device(device))
    n_nodes = A_norm.shape[0]
    n_integrate_steps = int(np.round(T / dt)) + 1

    # Torch tensors on device
    start = time()
    A_norm = torch.tensor(A_norm.astype("float32"), **devp)
    x0 = torch.tensor(x0.reshape(-1, 1).astype("float32"), **devp)
    xf = torch.tensor(xf.reshape(-1, 1).astype("float32"), **devp)
    xr = torch.zeros((n_nodes, 1), **devp)

    I = torch.eye(n_nodes, **devp)
    B = I if B is None else torch.tensor(np.array(B).astype("float32"), **devp)
    S = I if S is None else torch.tensor(np.array(S).astype("float32"), **devp)
    T, rho, dt = torch.tensor([T, rho, dt], **devp)

    M = torch.cat((torch.cat((A_norm, (-B @ B.T) / (2 * rho)), dim=1),
                   torch.cat((-2 * S, -A_norm.T), axis=1)))

    c = torch.cat([torch.zeros((n_nodes, 1), **devp), 2 * S @ xr], axis=0)

    # Ignore MPS fallback warning, still faster on MPS than CPU
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = torch.linalg.solve(M, c)

    E = torch.linalg.matrix_exp(M * T)
    E11 = E[:n_nodes][:, :n_nodes]
    E12 = E[:n_nodes][:, n_nodes:]

    dd = xf - (E11 @ x0)  - (torch.cat([E11 - I, E12], axis=1) @ c)

    # Ignore MPS fallback warning, still faster on MPS than CPU
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        l0 = torch.linalg.solve(E12, dd)

    big_I = torch.eye(2 * n_nodes, **devp)
    z = torch.zeros((2 * n_nodes, n_integrate_steps), **devp)
    z[:, 0:1] = torch.cat([x0, l0], axis=0)

    Ad = torch.linalg.matrix_exp(M * dt)
    Bd = ((Ad - big_I) @ c).flatten()

    for i in np.arange(1, n_integrate_steps):
        z[:, i] = Ad @ z[:, i - 1] + Bd

    x = z[:n_nodes, :]
    u = (- B.T @ z[n_nodes:, :]) / (2 * rho)
    E = torch.sum(u ** 2)

    err_costate = torch.linalg.norm(E12 @ l0 - dd)
    err_xf = torch.linalg.norm(x[:, -1].reshape(-1, 1) - xf)

    return float(E), x.T.cpu().numpy(), u.T.cpu().numpy(), [float(err_costate), float(err_xf)]


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


def get_cti_block(A_norms, x0s, xfs, T=1, rho=1, dt = 0.001, device="cpu", ret_mem=False, **extras):
    """ """

    assert len(A_norms) == len(x0s) or (len(A_norms.shape) == 2)

    M, E11, E12, Ad, S, B_T, dd_op = get_cti_A_components(A_norms, T=T, rho=rho, dt=dt, device=device)

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
    n_batch = max(min(get_max_batch_size(n_nodes, device), 200), 1)
    n_blocks = int(np.ceil(n_states / n_batch))
    n_integrate_steps = int(np.round(T / dt)) + 1

    multiple_A = len(A_norms.shape) == 3
    E_s = np.empty(n_states)
    x_s = np.empty((n_states, n_integrate_steps, n_nodes))
    u_s = np.empty((n_states, n_integrate_steps, n_nodes))
    err_s = np.empty((n_states, 2))

    for i in tqdm(range(n_blocks), leave=False, desc="Single trajectory set"):
        sl = slice(n_batch * i, min(n_batch * (i + 1), n_states))
        A_block = A_norms[sl] if multiple_A else A_norms
        block_results = get_cti_block(A_block, x0s[sl], xfs[sl], T=T, dt=dt, rho=rho, device=device.type)
        E_s[sl], x_s[sl], u_s[sl], err_s[sl] = block_results

    return E_s, x_s, u_s, err_s


def calc_trajectories(subjects, A_set, ntf_array, subj_save_path):
    """ """

    pbar = tqdm(total=len(A_set))
    for subj, AT_i, tf_array_i in zip(subjects, A_set, ntf_array):
        pbar.set_description(f"Calculating energy trajectories ({subj})")
        x0s, xfs = tf_array_i[:-1], tf_array_i[:-1]
        E_s, x_s, u_s, err_s = get_cti_batch(AT_i, x0s, xfs, device="mps")

        results_obj = [E_s, np.mean(u_s, axis=1), np.mean(u_s ** 2, axis=1), err_s]
        with open(subj_save_path.format(subject=subj), 'wb') as file:
            results_i = pickle.dump(results_obj, file)
        pbar.update(1)


def write_control_pkl(pkl_path, subjects, A_set, ntf_array):
    """ """
    assert len(subjects) == len(A_set)
    assert len(A_set) == len(ntf_array)

    with open(pkl_path, "wb") as file:
        pickle.dump((subjects, A_set, ntf_array), file)


def load_control_pkl(pkl_path):
    """ """
    with open(pkl_path, 'rb') as file:
        subjects, A_set, ntf_array = pickle.load(file)

    assert len(subjects) == len(A_set)
    assert len(A_set) == len(ntf_array)
    return subjects, A_set, ntf_array


def calculate_control_pkl(control_pkl_path: str, pkl_path_uf: str):
    """ """

    assert os.path.exists(control_pkl_path)
    assert "{subject}" in pkl_path_uf
    print("Loading control pkl: ...", end='\r')
    subjects, A_set, ntf_array = load_control_pkl(control_pkl_path)
    print("Loading control pkl: done")
    calc_trajectories(subjects, A_set, ntf_array, pkl_path_uf)


def norm(A, c=1):
    A = torch.tensor(A.astype("float32"), device="cpu")
    c = torch.tensor(c, device="cpu")
    w, _ = torch.linalg.eigh(A)
    l = torch.abs(w).max()

    # Matrix normalization for discrete-time systems
    A_norm = A / (c + l)
    A_norm = A_norm - torch.eye(A.shape[0], device="cpu")

    return A_norm.cpu().numpy()


def machine_tests():
    """ """
    print("Testing t_energies")
    n_nodes = 400
    n_batch = get_max_batch_size(n_nodes, device=get_device())

    A_set = np.random.randn(n_batch, n_nodes, n_nodes)
    A_norms = np.array([norm(A_i) for A_i in tqdm(A_set, desc="norming As")])

    x0s = np.array([x / np.linalg.norm(x) for x in np.random.randn(n_batch, n_nodes)])
    xfs = np.array([x / np.linalg.norm(x) for x in np.random.randn(n_batch, n_nodes)])

    if n_batch <= 10:
        print("\nCPU non-batched:")
        start = time()
        [get_cti_torch(*args, device="cpu") for args in zip(A_norms, x0s, xfs)];
        print(f"CPU non-batched time: {time() - start: 0.2f}s")

        print("\nCPU Batch run:")
        start = time()
        get_cti_block(A_norms, x0s, xfs, device="cpu")
        print(f"CPU Batch time: {time() - start: 0.2f}s")

    if torch.backends.mps.is_available():
        print("\nMPS Batch run:")
        start = time()
        get_cti_block(A_norms, x0s, xfs, device="mps")
        print(f"MPS Batch time: {time() - start: 0.2f}s")

        print("\nMPS non-batched:")
        start = time()
        [get_cti_torch(*args, device="mps") for args in zip(A_norms, x0s, xfs)];
        print(f"MPS non-batched time: {time() - start: 0.2f}s")

    if torch.cuda.is_available():
        print("\nCUDA Batch run:")
        start = time()
        get_cti_block(A_norms, x0s, xfs, device="cuda")
        delta = time() - start
        print(f"CUDA Batch time: {delta: 0.2f}s ({n_batch / delta:0.2f})\n")

        print("\nCUDA non-batched:")
        start = time()
        [get_cti_torch(*args, device="cuda") for args in zip(A_norms, x0s, xfs)];
        delta = time() - start
        print(f"CUDA non-batched time: {delta: 0.2f}s ({n_batch / delta:0.2f})\n")



def get_arguments():
    """ """
    test_mode = ("--test" in sys.argv) or ("-t" in sys.argv)
    calc_mode = not test_mode

    parser = argparse.ArgumentParser(prog='t_energies', description='')
    parser.add_argument('-t', "--test", dest='test', action="store_true", required=False,
                        help="Txt file with paths of cifti files or cifti glob path")
    parser.add_argument('-c', "--control-pkl", dest='control_pkl', action="store", type=str,
                        required=calc_mode, help="control pkl path")
    parser.add_argument('-o', "--out", dest='out_dir', action="store", type=str, default="",
                        required=False, help="output directory path")
    args = parser.parse_args()

    return args


def main():
    args = get_arguments()
    if args.test:
        machine_tests()

    else:
        assert os.path.exists(args.out_dir) or args.out_dir == ""
        assert os.path.exists(args.control_pkl)

        out_path = os.path.join(args.out_dir, "{subject}_tfMRI_CTI.pkl")
        calculate_control_pkl(args.control_pkl, out_path)


if __name__ == '__main__':
    main()
