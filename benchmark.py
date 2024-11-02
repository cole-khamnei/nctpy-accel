import jax_energies as je
import t_energies as te

import nctpy
from nctpy import utils as nct_utils
from nctpy import energies as nct_e

import numpy as np

from tqdm.auto import tqdm

NP_CLOSE_PARAMS = dict(rtol=1e-05, atol=1e-05)


def array_equal(A, B, show=True, rtol=1e-05, atol=1e-05):
    """ """
    close = np.isclose(A, B, rtol=rtol, atol=atol)
    value = close.all()
    if show and not value:
        print(A - B)
        print(close)

        print(np.prod(A.shape) - np.sum(close * 1), (1 - np.mean(close * 1)), "wrong values")
    return value


def get_relative_error(A, B):
    """ """
    return np.abs(A - B) / (A + (A == 0) + 1)


def get_random_states(n_nodes, n_batch):
    """ """
    x0s = np.array([x / np.linalg.norm(x) for x in np.random.randn(n_batch, n_nodes)])
    xfs = np.array([x / np.linalg.norm(x) for x in np.random.randn(n_batch, n_nodes)])

    return x0s, xfs


def get_random_A_set(n_batch, n_nodes):
    """ """
    A_rand_set = np.random.randn(n_batch, n_nodes, n_nodes)
    A_rand_set = np.array([np.tril(a) + np.tril(a, -1).T for a in A_rand_set])

    return A_rand_set


def get_random_A_norms(n_batch, n_nodes, system="continuous"):
    """ """
    A_set = get_random_A_set(n_batch, n_nodes)
    return np.array([te.matrix_norm(A_i, c=1, system=system) for A_i in A_set])


def test_matrix_norms():
    """ """
    n_batch = 10
    n_nodes = 400
    A_set = get_random_A_set(n_batch, n_nodes)


    system = "continuous"
    c = 1

    for system in te.VALID_SYSTEMS:
        for c in tqdm([1, 1.5, 3], desc="Testing matrix_norm with various c's:", leave=False):
            normed_A_base = np.array([nct_utils.matrix_normalization(A_i, c=c, system=system) for A_i in A_set])
            normed_A_torch = np.array([te.matrix_norm(A_i, c=c, system=system) for A_i in A_set])
            assert array_equal(normed_A_base, normed_A_torch)

            normed_A_jax = np.array([je.matrix_norm(A_i, c=c, system=system) for A_i in A_set])
            assert array_equal(normed_A_base, normed_A_jax, show=True)


def test_matrix_norm_speed():
    """ """
    n_nodes = 400
    n_batch = 20
    A_set = get_random_A_set(n_batch, n_nodes)
    [nct_utils.matrix_normalization(A_i, system="continuous") for A_i in tqdm(A_set, desc="base")]

    n_batch = 500
    A_set = get_random_A_set(n_batch, n_nodes)
    [te.matrix_norm(A_i) for A_i in tqdm(A_set, desc="torch")]
    [je.matrix_norm(A_i) for A_i in tqdm(A_set, desc="jax")]


def test_cti_accuracy(backend_str):
    """ """
    n_nodes = 400
    system = "continuous"
    T = 1

    if backend_str == "torch":
        backend = te
    elif backend_str == "jax":
        backend = je
    else:
        raise ValueError(f"Invalid backend '{backend_str}'")

    n_batch = te.get_max_batch_size(n_nodes, device=te.get_device())

    x0s, xfs = get_random_states(n_nodes, n_batch)
    A_norms = get_random_A_norms(n_batch, n_nodes, system=system)

    B = np.eye(n_nodes)

    x_errors, u_errors = [], []
    for A_norm, x0, xf in zip(A_norms, x0s, xfs):
        E_b, x_b, u_b, err_b = backend.get_control_inputs(A_norm, x0, xf, T=T)
        x, u, err = nct_e.get_control_inputs(A_norm, T, B, x0, xf, system=system)

        assert array_equal(x, x_b, rtol=1e-4, atol=1e-4)
        assert array_equal(u, u_b, rtol=1e-4, atol=1e-4)

        x_errors.append(get_relative_error(x, x_b).ravel())
        u_errors.append(get_relative_error(u, u_b).ravel())

        # Error values not accurate
        # for i in range(len(err_t)):
        #     print(err, err_t)
        #     assert array_equal(err[i], err_t[i])
    print(f"{backend_str.upper()} CTI Values: accurate")
    print(f"\tX Error: {np.mean(x_errors) * 100:0.5f}%")
    print(f"\tU Error: {np.mean(u_errors) * 100:0.5f}%\n")


def test_cti_single_event_speed():
    """ """
    n_nodes = 400
    system = "continuous"
    T = 1


    n_batch = te.get_max_batch_size(n_nodes, device=te.get_device())
    n_batch = n_batch * 10

    x0s, xfs = get_random_states(n_nodes, n_batch)
    A_norms = get_random_A_norms(n_batch, n_nodes, system=system)

    B = np.eye(n_nodes)
    pbar = tqdm(total=len(x0s), desc="NCTPY      ")
    for A_norm, x0, xf in zip(A_norms, x0s, xfs):
        nct_e.get_control_inputs(A_norm, T, B, x0, xf, system=system)
        pbar.update(1)
    
    n_batch = n_batch * 3
    x0s, xfs = get_random_states(n_nodes, n_batch)
    A_norms = get_random_A_norms(n_batch, n_nodes, system=system)

    pbar = tqdm(total=len(x0s), desc="NCT-TORCH")
    for A_norm, x0, xf in zip(A_norms, x0s, xfs):
        te.get_control_inputs(A_norm, x0, xf)
        pbar.update(1)

    pbar = tqdm(total=len(x0s), desc="NCT-JAX  ")
    for A_norm, x0, xf in zip(A_norms, x0s, xfs):
        je.get_control_inputs(A_norm, x0, xf)
        pbar.update(1)


def test_cti_block_accuracy(backend_str):
    """ """

    n_nodes = 400
    n_batch = te.get_max_batch_size(n_nodes, device=te.get_device())

    x0s, xfs = get_random_states(n_nodes, n_batch)
    A_norms = get_random_A_norms(n_batch, n_nodes)

    j_outs = je.get_cti_block(A_norms, x0s, xfs)
    t_outs = te.get_cti_block(A_norms, x0s, xfs)

    for j_out, t_out in zip(j_outs, t_outs):
        assert array_equal(j_out, t_out, rtol=1e-3, atol=1e-4)


def test_cti_block_speed():
    """ """
    n_nodes = 400
    n_batch = te.get_max_batch_size(n_nodes, device=te.get_device())
    print(n_batch)
    x0s, xfs = get_random_states(n_nodes, n_batch)
    A_norms = get_random_A_norms(n_batch, n_nodes)

    n_reps = 100
    for _ in tqdm(range(n_reps), desc="testing torch cti block"):
        te.get_cti_block(A_norms, x0s, xfs, device="cuda")

    for _ in tqdm(range(n_reps), desc="testing jax cti block"):
        je.get_cti_block(A_norms, x0s, xfs)


def main():
    """ """
    print("Running benchmark tests:\n")
    # test_cti_accuracy("torch")
    # test_cti_accuracy("jax")
    # test_cti_block_accuracy("jax")
    test_cti_block_speed()
    # test_cti_single_event_speed()


    # tests = [test_matrix_norms]
    # for test in tests:
    #     print(f"Running {test.__name__}:")
    #     test()


if __name__ == '__main__':
    main()
