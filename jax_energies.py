import numpy

import jax
import jax.numpy as jnp

from tqdm.auto import tqdm
from . import utils


# ------------------------------------------------------------------- #
# --------------------    JAX Precompile LUT     -------------------- #
# ------------------------------------------------------------------- #
"""
Fairly certain there is a better way to do this with JAX, as it auto-registers functions
with hash, tried @partial with staticargs which worked, but this was quicker to implement.

May change in future.

"""

class CompiledFunctionSet():
    def __init__(self, make_function, initial_args=[], make_key=None):
        self.make_function = make_function
        self.make_key = make_key or (lambda args: "-".join(str(a) for a in args))
        self.compiled_versions = {self.make_key(args): self.make_function(*args) for args in initial_args}

    def __call__(self, *args):
        key = self.make_key(args)
        if key not in self.compiled_versions:
            self.compiled_versions[key] = self.make_function(*args)
        return self.compiled_versions[key]


# ------------------------------------------------------------------- #
# --------------------     JAX NCTPY Utils      -------------------- #
# ------------------------------------------------------------------- #

@jax.jit
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return jnp.allclose(a, a.T, rtol=rtol, atol=atol)


@jax.jit
def eigh_norm(A, c):
    """ """
    w, _ = jnp.linalg.eigh(A)
    return A / (c + jnp.abs(w).max())


def eig_norm(A, c):
    """ """
    w, _ = numpy.linalg.eig(A)
    return A / (c + numpy.abs(w).max())


def matrix_norm(A, c=1, system="continuous", symmetric=False):
    """ """
    if A.ndim > 2:
        return numpy.array([matrix_norm(A_i, c=1, system=system, symmetric=symmetric) for A_i in A])

    utils.check_system(system)

    # eig_normed_A = eig_norm(A, c)
    eig_normed_A = eigh_norm(A, c) if check_symmetric(A) or symmetric else eig_norm(A, c)
    return eig_normed_A - jnp.eye(A.shape[0]) if system == 'continuous' else eig_normed_A


# ------------------------------------------------------------------- #
# ----------------   Control Input Build Functions    --------------- #
# ------------------------------------------------------------------- #


def build_compute_dynamics_matrices(n_nodes, n_integrate, n_batch, n_A):
    """ """
    xr = jnp.zeros((n_nodes, 1))
    concat_dim = 2 if n_A > 0 else 1
    to_batch = lambda *tensors: (jnp.tile(jnp.expand_dims(t_i, 0), (n_A, 1, 1)) for t_i in tensors)

    def compute_dynamics_matrices(A_norm, S, B, T, dt, rho):

        B_T = B
        I, big_I, zero_array = jnp.eye(n_nodes), jnp.eye(2 * n_nodes), jnp.zeros((n_nodes, 1))
        S_xr = S @ xr

        if n_A > 0:
            B, B_T, S, I, S_xr, zero_array = to_batch(B, B_T, S, I, S_xr, zero_array)
            A_norm_T = A_norm.transpose(0, 2, 1)
        else:
            A_norm_T = A_norm.T

        M = jnp.concatenate((jnp.concatenate((A_norm, (-B @ B_T) / (2 * rho)), axis=concat_dim),
                             jnp.concatenate((-2 * S, -A_norm_T), axis=concat_dim)),
                             axis=concat_dim - 1)

        c = jnp.concatenate([zero_array, 2 * S_xr], axis=concat_dim - 1)
        c = jnp.linalg.solve(M, c)

        Ad = jax.scipy.linalg.expm(M * dt)
        Bd = ((Ad - big_I) @ c).squeeze()
        if n_batch > 0 and n_A == 0:
            Ad = jnp.tile(Ad, (n_batch, 1, 1))

        E = jax.scipy.linalg.expm(M * T)
        if n_A > 0:
            E11, E12 = E[:, :n_nodes, :n_nodes], E[:, :n_nodes, n_nodes:]
        else:
            E11, E12 = E[:n_nodes][:, :n_nodes], E[:n_nodes][:, n_nodes:]

        dd_bias = jnp.concatenate([E11 - I, E12], axis=concat_dim) @ c
        return Ad, Bd, E11, E12, dd_bias, B_T

    return jax.jit(compute_dynamics_matrices)

_compute_dynamics_matrices_funcs = CompiledFunctionSet(build_compute_dynamics_matrices)


def build_compute_single_trajectory(n_nodes, n_integrate):
    """ """
    I = jnp.eye(n_nodes)
    def compute_single_trajectory(A_norm, dynamics_matrices, x0, xf, rho):

        Ad, Bd, E11, E12, dd_bias, B_T = dynamics_matrices

        x0 = jnp.array(x0.reshape(-1, 1))
        xf = jnp.array(xf.reshape(-1, 1))

        dd = xf - (E11 @ x0) - dd_bias
        l0 = jnp.linalg.solve(E12, dd)

        z = jnp.zeros((2 * n_nodes, n_integrate))
        def integrate_step(i, z):
            return z.at[:, i].set(Ad @ z[:, i - 1] + Bd)

        z = z.at[:, :1].set(jnp.concatenate([x0, l0], axis=0))
        z = jax.lax.fori_loop(1, n_integrate, integrate_step, z)

        x = z[:n_nodes, :]
        u = (- B_T @ z[n_nodes:, :]) / (2 * rho)
        E = jnp.sum(u ** 2)

        err_costate = jnp.linalg.norm(E12 @ l0 - dd)
        err_xf = jnp.linalg.norm(x[:, -1].reshape(-1, 1) - xf)
        err = [jnp.array(err_costate, float), jnp.array(err_xf, float)]

        return E, x.T, u.T, err

    return jax.jit(compute_single_trajectory)

_compute_single_trajectory_funcs = CompiledFunctionSet(build_compute_single_trajectory)


# ------------------------------------------------------------------- #
# --------------------    Block Trajectory       -------------------- #
# ------------------------------------------------------------------- #


def build_compute_block_trajectory(n_nodes, n_batch, n_integrate):
    """ """
    state_tensor_shape = (n_batch, n_nodes, 1)
    big_I = jnp.eye(2 * n_nodes)

    def compute_block_trajectory(A_norms, dynamics_matrices, x0s, xfs, rho):
        """ """
        Ad, Bd, E11, E12, dd_bias, B_T = dynamics_matrices

        x0s_b = jnp.array(x0s.reshape(state_tensor_shape))
        xfs_b = jnp.array(xfs.reshape(state_tensor_shape))

        dd = xfs_b - (E11 @ x0s_b) - dd_bias
        l0 = jnp.linalg.solve(E12, dd)

        z = jnp.zeros((n_integrate, n_batch, 2 * n_nodes))
        z0 = jnp.concatenate([x0s_b, l0], axis=1).squeeze()
        z = z.at[0].set(z0)

        def integrate_step(i, z):
            z_i = (Ad @ jnp.expand_dims(z[i - 1], -1)).squeeze() + Bd
            return z.at[i].set(z_i)

        z = jax.lax.fori_loop(1, n_integrate, integrate_step, z)
        z = z.transpose(1, 2, 0)

        x = z[:, :n_nodes, :]
        u = (- B_T @ z[:, n_nodes:, :]) / (2 * rho)
        E = jnp.trapezoid(u ** 2, axis=1).sum(axis=1)

        err_xf = jnp.linalg.norm(x[:, :, -1:] - xfs_b, axis=1)
        err_costate = jnp.linalg.norm(E12 @ l0 - dd, axis=1)
        err = jnp.concatenate([err_xf, err_costate], axis=1)

        x = x.transpose(0, 2, 1)
        u = u.transpose(0, 2, 1)

        return E, x, u, err

    return jax.jit(compute_block_trajectory)

_compute_block_trajectory_funcs = CompiledFunctionSet(build_compute_block_trajectory)


# ------------------------------------------------------------------- #
# --------------------    Control Input Func     -------------------- #
# ------------------------------------------------------------------- #


def get_CTI_dimensions(A_norm, x0s, xfs, T, dt):
    """ """
    assert x0s.shape == xfs.shape
    assert x0s.shape[-1] == A_norm.shape[-1]

    n_nodes = x0s.shape[-1]
    n_integrate = int(numpy.round(T / dt) + 1)
    n_states = len(x0s) if x0s.ndim == 2 else 0
    n_A = A_norm.shape[0] if A_norm.ndim == 3 else 0


    return n_nodes, n_integrate, n_states, n_A


def get_S_and_B_matrices(S, B, n_nodes):
    """ """
    I = jnp.eye(n_nodes)
    B = I if B is None else jnp.array(B)
    S = I if S is None else jnp.array(S)
    assert S.shape == B.shape

    return S, B


def get_control_inputs(A_norm, x0s, xfs, B=None, S=None, T=1, dt=0.001, rho=1, dynamics_matrices=None):
    """ """

    n_nodes, n_integrate, n_batch, n_A  = get_CTI_dimensions(A_norm, x0s, xfs, T, dt)
    S, B = get_S_and_B_matrices(S, B, n_nodes)

    compute_dynamics_matrices = _compute_dynamics_matrices_funcs(n_nodes, n_integrate, n_batch, n_A)
    dynamics_matrices = compute_dynamics_matrices(A_norm, S, B, T, dt, rho)

    if n_batch == 0:
        compute_trajectory = _compute_single_trajectory_funcs(n_nodes, n_integrate)
    else:
        compute_trajectory = _compute_block_trajectory_funcs(n_nodes, n_batch, n_integrate)

    return compute_trajectory(A_norm, dynamics_matrices, x0s, xfs, rho)


def get_cti_batch(A_norms, x0s, xfs, B=None, S=None, T=1, dt=0.001, rho=1, n_batch=20, pbar_kws=dict(leave=False)):
    """ """
    #TODO: get from function for system specifics for batch size

    multiple_A = A_norms.ndim == 3
    assert len(A_norms) == len(x0s) or not multiple_A

    n_nodes, n_integrate, n_states, n_A  = get_CTI_dimensions(A_norms, x0s, xfs, T, dt)
    n_blocks = int(numpy.ceil(n_states / n_batch))
    S, B = get_S_and_B_matrices(S, B, n_nodes)

    E_s = numpy.empty(n_states)
    x_s = numpy.empty((n_states, n_integrate, n_nodes))
    u_s = numpy.empty((n_states, n_integrate, n_nodes))
    err_s = numpy.empty((n_states, 2))


    compute_dynamics_matrices = _compute_dynamics_matrices_funcs(n_nodes, n_integrate, n_batch, n_A)
    if not multiple_A:
        single_A_dynamics_matrices = compute_dynamics_matrices(A_norms, S, B, T, dt, rho)

    pbar = tqdm(total=n_states, desc="JAX CTI trajectory set", **pbar_kws)
    for i in range(n_blocks):
        sl = slice(n_batch * i, min(n_batch * (i + 1), n_states))

        n_batch_i = n_states % n_batch if i == (n_blocks - 1) else n_batch
        if multiple_A:
            dynamics_matrices = compute_dynamics_matrices(A_norms[sl], S, B, T, dt, rho)
        else:
            Ad, Bd, E11, E12, dd_bias, B_T = single_A_dynamics_matrices
            dynamics_matrices = (Ad[:n_batch_i], Bd, E11, E12, dd_bias, B_T) # batch size effects of Ad

        compute_trajectory = _compute_block_trajectory_funcs(n_nodes, n_batch_i, n_integrate)
        trajectory_results = compute_trajectory(A_norms[sl], dynamics_matrices, x0s[sl], xfs[sl], rho)
        E_s[sl], x_s[sl], u_s[sl], err_s[sl] = trajectory_results

        pbar.update(sl.stop - sl.start)
    pbar.close()

    return E_s, x_s, u_s, err_s


# ------------------------------------------------------------------- #
# --------------------            End            -------------------- #
# ------------------------------------------------------------------- #
