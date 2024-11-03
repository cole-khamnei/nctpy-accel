import jax_energies as je
import t_energies as te

import nctpy
from nctpy import utils as nct_utils
from nctpy import energies as nct_e

import numpy as np

from utils import Timer
from tqdm.auto import tqdm

NP_CLOSE_PARAMS = dict(rtol=1e-05, atol=1e-05)


def array_equal(A, B, show=True, rtol=1e-05, atol=1e-05):
    """ """
    assert A.shape == B.shape, f"A-shape: {A.shape} and B-shape: {B.shape}"
    close = np.isclose(A, B, rtol=rtol, atol=atol)
    value = close.all()
    if show and not value:
        diff = A - B
        rel_diff = np.abs(diff / (A + (A == 0))) * 100
        print("A:", A)
        print("B:", B)
        print(diff)
        print(rel_diff)
        print(close)
        print(rel_diff.mean(axis=0).shape, rel_diff.mean(axis=0))
        # print(close.mean(axis=1).shape, close.mean(axis=1))
        print("Average absolute difference (%):", np.mean(np.abs(diff)))
        print("Average relative difference (%):", np.mean(rel_diff))
        print(np.prod(A.shape) - np.sum(close * 1), (1 - np.mean(close * 1)), "wrong values")
    return value


def get_relative_error(A, B):
    """ """
    return np.abs(A - B) / (A + (A == 0) + 1)


def get_random_states(n_nodes, n_batch, seed=1):
    """ """
    np.random.seed(seed)
    x0s = np.array([x / np.linalg.norm(x) for x in np.random.randn(n_batch, n_nodes) * 100])
    xfs = np.array([x / np.linalg.norm(x) for x in np.random.randn(n_batch, n_nodes) * 100])

    return x0s, xfs


def get_random_A_set(n_batch, n_nodes, seed=1):
    """ """
    np.random.seed(seed)
    A_rand_set = np.random.randn(n_batch, n_nodes, n_nodes) * 1000
    A_rand_set = np.array([np.tril(a) + np.tril(a, -1).T for a in A_rand_set])

    return A_rand_set


def get_random_A_norms(n_batch, n_nodes, seed=1, system="continuous"):
    """ """
    np.random.seed(seed)
    A_set = get_random_A_set(n_batch, n_nodes, seed=seed)
    return np.array([te.matrix_norm(A_i, c=1, system=system) for A_i in A_set])


def get_NCT_args_set(n_batch, n_nodes, seed = 1, system = "continuous"):
    """ """
    A_norms = get_random_A_norms(n_batch, n_nodes, seed=seed, system=system)
    x0s, xfs = get_random_states(n_nodes, n_batch, seed=seed)
    return A_norms, x0s, xfs


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

    n_batch = 10
    x0s, xfs = get_random_states(n_nodes, n_batch)
    A_norms = get_random_A_norms(n_batch, n_nodes, system=system)

    B = np.eye(n_nodes)

    x_errors, u_errors = [], []
    pbar = tqdm(total=len(x0s), desc=f"Checking {backend.__name__} control inputs accuracy")
    for A_norm, x0, xf in zip(A_norms, x0s, xfs):
        E_b, x_b, u_b, err_b = backend.get_control_inputs(A_norm, x0, xf, T=T)
        x, u, err = nct_e.get_control_inputs(A_norm, T, B, x0, xf, system=system)

        assert array_equal(x, x_b, rtol=1e-3, atol=1e-4)
        assert array_equal(u, u_b, rtol=1e-3, atol=1e-4)

        x_errors.append(get_relative_error(x, x_b).ravel())
        u_errors.append(get_relative_error(u, u_b).ravel())

        ## TODO: Add in error checks    
        # Error values not accurate
        # for i in range(len(err_t)):
        #     print(err, err_t)
        #     assert array_equal(err[i], err_t[i])
        pbar.update(1)

    pbar.close()
    print(f"{backend_str.upper()} CTI Values: accurate")
    print(f"\tX Error: {np.mean(x_errors) * 100:0.5f}%")
    print(f"\tU Error: {np.mean(u_errors) * 100:0.5f}%\n")


def test_cti_single_event_speed():
    """ """
    n_nodes = 400
    B = np.eye(n_nodes)
    system = "continuous"
    T = 1

    n_batch = 20
    A_norms, x0s, xfs = get_NCT_args_set(n_batch, n_nodes, seed=1)
    
    print("Testing single A, single state speeds:")
    with Timer("JAX precompilation:") as t:
        je.get_control_inputs(A_norms[0], x0s[0], xfs[0])
    print()

    pbar = tqdm(total=len(x0s), desc="NCTPY    ")
    for A_norm, x0, xf in zip(A_norms, x0s, xfs):
        nct_e.get_control_inputs(A_norm, T, B, x0, xf, system=system)
        pbar.update(1)
    
    n_batch = n_batch * 5
    A_norms, x0s, xfs = get_NCT_args_set(n_batch, n_nodes, seed=1)

    pbar = tqdm(total=len(x0s), desc="NCT-TORCH")
    for A_norm, x0, xf in zip(A_norms, x0s, xfs):
        te.get_control_inputs(A_norm, x0, xf, device=te.get_device())
        pbar.update(1)
    pbar.close()

    pbar = tqdm(total=len(A_norms), desc="NCT-JAX  ")
    for A_norm, x0, xf in zip(A_norms, x0s, xfs):
        je.get_control_inputs(A_norms[0], x0, xf)
        pbar.update(1)
    pbar.close()
    print()


def test_cti_block_accuracy():
    """ """
    n_nodes = 400
    n_batch = te.get_max_batch_size(n_nodes, device=te.get_device())
    print(f"Testing JAX & Torch Block Methods Accuracy:\n  Batch Size = {n_batch}")

    i = 0
    for A_si, label in zip([0, slice(None, None)], ["single A", "multiple A"]):
        A_norms, x0s, xfs = get_NCT_args_set(n_batch, n_nodes, seed=32)
        j_outs = je.get_control_inputs(A_norms[A_si], x0s, xfs)
        t_outs = te.get_cti_block(A_norms[A_si], x0s, xfs)

        for j_out, t_out in zip(j_outs, t_outs):
            assert array_equal(j_out, t_out, rtol=1e-3, atol=1e-3)
        print(f"  Block method with {label}: accurate.")


def test_cti_block_speed():
    """ """
    print(f"Testing JAX & Torch Block Methods Speed Comparison:")
    n_nodes = 400
    n_samples = 2500

    n_batch = 80
    n_reps = n_samples // n_batch
    A_norms, x0s, xfs = get_NCT_args_set(n_batch, n_nodes, seed=32)

    with Timer() as t:
        je.get_control_inputs(A_norms[0], x0s, xfs)
    

    for A_si, A_label in zip([0, slice(None, None)], ["single", "multiple"]):

        pbar = tqdm(total=n_reps * n_batch, desc=f"NCT-JAX   ({A_label} A | multiple states | batch_size: {n_batch}):")
        for _ in range(n_reps):
            je.get_control_inputs(A_norms[A_si], x0s, xfs)
            pbar.update(n_batch)
        pbar.close()

        n_batch = te.get_max_batch_size(n_nodes, device=te.get_device())
        # n_batch = 30
        n_reps = n_samples // n_batch
        A_norms, x0s, xfs = get_NCT_args_set(n_batch, n_nodes, seed=32)
        
        pbar = tqdm(total=n_reps * n_batch, desc=f"NCT-Torch ({A_label} A | multiple states | batch_size: {n_batch}):")
        for _ in range(n_reps):
            te.get_cti_block(A_norms[A_si], x0s, xfs, device="cuda")
            pbar.update(n_batch)
        pbar.close()


def main():
    """ """
    print("\nRunning Accelerated NCT validation tests:")
    test_cti_accuracy("torch")
    test_cti_accuracy("jax")
    test_cti_single_event_speed()
    test_cti_block_accuracy()
    test_cti_block_speed()


if __name__ == '__main__':
    main()