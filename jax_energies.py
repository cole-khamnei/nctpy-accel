import jax
import jax.numpy as jnp
import numpy

import time

VALID_SYSTEMS = ["discrete", "continuous"]

class Timer:
    def __init__(self, label=""):
        self.label = label

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.duration = self.end - self.start
        # print(f"Duration {self.label}: {self.duration:0.5f} seconds")


@jax.jit
def matrix_dnorm(A, c):
    """ """
    w, _ = jnp.linalg.eigh(A)
    l = jnp.abs(w).max()
    return A / (c + l)

@jax.jit
def matrix_cnorm(A, c):
    """ """
    return matrix_dnorm(A, c) - jnp.eye(A.shape[0])


def matrix_norm(A, c=1, system="continuous"):
    """ """
    assert system in VALID_SYSTEMS, f"Invalid system '{system}', valid sytems: {VALID_SYSTEMS}"

    if system == 'continuous':
        return matrix_cnorm(A, c)
    else:
        return matrix_dnorm(A, c)

@jax.jit
def nct_trajectory_intermediates(A_norm, B, S, T, dt, rho, x0, xf, xr, n_nodes, I, big_I, zero_array):
    """ """

    M = jnp.concatenate((jnp.concatenate((A_norm, (-B @ B.T) / (2 * rho)), axis=1),
                             jnp.concatenate((-2 * S, -A_norm.T), axis=1)))
    c = jnp.concatenate([zero_array, 2 * S @ xr], axis=0)
    c = jnp.linalg.solve(M, c)

    E = jax.scipy.linalg.expm(M * T)

    n_nodes = len(A_norm)
    E11 = E[:n_nodes][:, :n_nodes]
    E12 = E[:n_nodes][:, n_nodes:]

    dd = xf - (E11 @ x0)  - (jnp.concatenate([E11 - I, E12], axis=1) @ c)
    l0 = jnp.linalg.solve(E12, dd)
    Ad = jax.scipy.linalg.expm(M * dt)
    Bd = ((Ad - big_I) @ c).reshape(-1)
    return M, c, E12, dd, l0, Ad, Bd


@jax.jit
def nct_integrate_trajectory(z, x0, l0, Ad, Bd):
    """ """
    z = z.at[:, 0:1].set(jnp.concatenate([x0, l0], axis=0))
    for i in jnp.arange(1, 1001):
        z = z.at[:, i].set(Ad @ z[:, i - 1] + Bd)
    return z


def get_control_inputs(A_norm, x0, xf, B=None, S=None, T=1, rho=1, n_integrate_steps=1001):
    """ """
    with Timer("prep") as t:
        n_nodes = A_norm.shape[0]
        dt = jnp.array(T / (n_integrate_steps - 1))
        # n_integrate_steps = jnp.array(jnp.round(T / dt), int) + 1

        # jnp tensors on device
        A_norm = jnp.array(A_norm)
        x0 = jnp.array(x0.reshape(-1, 1))
        xf = jnp.array(xf.reshape(-1, 1))
        xr = jnp.zeros((n_nodes, 1))

        I = jnp.eye(n_nodes)
        big_I = jnp.eye(2 * n_nodes)
        B = I if B is None else jnp.array(B)
        S = I if S is None else jnp.array(S)
        T, rho, dt = jnp.array([T, rho, dt])
        zero_array = jnp.zeros((n_nodes, 1))
        z = jnp.zeros((2 * n_nodes, n_integrate_steps))

    with Timer("intermediates") as t:
        M, c, E12, dd, l0, Ad, Bd = nct_trajectory_intermediates(A_norm, B, S, T, dt, rho, x0, xf, xr,
                                                                 n_nodes, I, big_I, zero_array)
    
    with Timer("integrate") as t:
        z = nct_integrate_trajectory(z, x0, l0, Ad, Bd)
    
    with Timer("end") as t:
        x = z[:n_nodes, :]
        u = (- B.T @ z[n_nodes:, :]) / (2 * rho)
        E = jnp.sum(u ** 2)

        err_costate = jnp.linalg.norm(E12 @ l0 - dd)
        err_xf = jnp.linalg.norm(x[:, -1].reshape(-1, 1) - xf)
        err = [jnp.array(err_costate, float), jnp.array(err_xf, float)]

    return jnp.array(E, float), numpy.array(x.T), numpy.array(u.T), err
    


########################################## Block Matrices #########################################################


class CompiledFunctionSet():
    def __init__(self, make_function, initial_args, make_key=None):
        self.make_function = make_function
        self.make_key = make_key or (lambda args: "-".join(str(a) for a in args))
        self.compiled_versions = {self.make_key(args): self.make_function(*args) for args in initial_args}

    def __call__(self, *args):
        key = self.make_key(args)
        if key not in self.compiled_versions:
            self.compiled_versions[key] = self.make_function(*args)
        return self.compiled_versions[key]


def build_cti_block_A_components(n_nodes, n_A):
    """ """
    I_b = jnp.eye(n_nodes)
    if n_A > 1:
        T_dims, cat_dim = (0, 2, 1), 2
        I_b = jnp.tile(I_b, (n_A, 1, 1))
    else:
        T_dims, cat_dim = (0, 1), 1
    S_b, B_b, B_b_T = I_b, I_b, I_b.transpose(*T_dims)

    def block_A_components_func(A_norm, T = 1, rho = 1, dt = 0.001):
        """ """
        A_norm_T = A_norm.transpose(*T_dims)
        M = jnp.concatenate([jnp.concatenate([A_norm, (-B_b @ B_b_T) / (2 * rho)], axis=cat_dim),
                             jnp.concatenate([- 2 * S_b, -A_norm_T], axis=cat_dim)],
                            axis=cat_dim - 1)
        E = jax.scipy.linalg.expm(M * T)
        Ad = jax.scipy.linalg.expm(M * dt)
        E11, E12 = E[:, :n_nodes][:, :, :n_nodes], E[:, :n_nodes][:, :, n_nodes:]
        dd_op = jnp.concatenate([E11 - I_b, E12], axis=cat_dim)
        return M, E11, E12, Ad, S_b, B_b_T, dd_op

    return jax.jit(block_A_components_func)

_CTI_block_A_component_funcs = CompiledFunctionSet(build_cti_block_A_components, [(400, 1)])


def build_cti_integrate(n_integrate_steps):
    """ """
    def cti_integrate_func(z, x0s_b, l0, Ad, Bd):
        z0 = jnp.concatenate([x0s_b, l0], axis=1).squeeze()
        z = z.at[0].set(z0)

        def body_fun(i, z):
            z_i = (Ad @ jnp.expand_dims(z[i - 1], 2) + Bd).squeeze()
            # return z
            return z.at[i].set(z_i)
        z = jax.lax.fori_loop(1, n_integrate_steps, body_fun, z)
        # jax.lax.fori_loop(1, n_integrate_steps, body_fun, z)
        return z

    return jax.jit(cti_integrate_func)

_CTI_integrate_funcs = CompiledFunctionSet(build_cti_integrate, [(1001,)], make_key=lambda n: n)


def build_cti_core(n_nodes, n_batch, n_A, n_integrate):
    
    create_block_A_components = _CTI_block_A_component_funcs(n_nodes, n_A)
    integrate_z = _CTI_integrate_funcs(n_integrate)
    state_tensor_shape = (n_batch, n_nodes, 1)
    
    def cti_core(A_norms, x0s, xfs, T, dt, rho):
        """ """
        M, E11, E12, Ad, S, B_T, dd_op = create_block_A_components(A_norms, T=T, rho=rho, dt=dt)
        
        x0s_b = jnp.array(x0s.reshape(state_tensor_shape))
        xfs_b = jnp.array(xfs.reshape(state_tensor_shape))
        xrs_b = jnp.zeros(state_tensor_shape)
        zero_array = jnp.zeros(state_tensor_shape)

        c = jnp.concatenate([zero_array, 2 * S @ xrs_b], axis=1)
        c = jnp.linalg.solve(M, c)
        
        dd = xfs_b - (E11 @ x0s_b) - (dd_op @ c)
        l0 = jnp.linalg.solve(E12, dd)

        big_I = jnp.eye(2 * n_nodes)
        Bd = (Ad - big_I) @ c

        z = jnp.zeros((n_integrate, n_batch, 2 * n_nodes))
        z = integrate_z(z, x0s_b, l0, Ad, Bd)
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

    return jax.jit(cti_core)


_CTI_core_funcs = CompiledFunctionSet(build_cti_core, [(400, 20, 20, 1001)])


def get_cti_block(A_norms, x0s, xfs, T=1, rho=1, dt = 0.001, intermediates=None):
    """ assumes multi A: TODO fix"""
    assert x0s.shape == xfs.shape

    n_batch, n_nodes = x0s.shape 
    n_A = A_norms.shape[0] if A_norms.ndim == 3 else 1
    n_integrate = int(numpy.round(T / dt) + 1)

    CTI_core = _CTI_core_funcs(n_nodes, n_batch, n_A, n_integrate)

    return CTI_core(A_norms, x0s, xfs, T, dt, rho)
