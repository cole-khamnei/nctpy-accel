import jax
import jax.numpy as jnp
import numpy

import time
from tqdm.auto import tqdm

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


def build_compute_dynamics_matrices(n_nodes, n_integrate, n_batch, n_A):

    xr = jnp.zeros((n_nodes, 1))
    concat_dim = 2 if n_A > 0 else 1
    to_batch = lambda *tensors: (jnp.tile(jnp.expand_dims(t_i, 0), (n_A, 1, 1)) for t_i in tensors)

    def compute_dynamics_matrices(A_norm, S, B, T, dt, rho):

        B_T = B
        I = jnp.eye(n_nodes)
        big_I = jnp.eye(2 * n_nodes)
        zero_array = jnp.zeros((n_nodes, 1))

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

        E = jax.scipy.linalg.expm(M * T)
        # E = jnp.linalg.matrix_power(Ad, n_integrate)

        if n_A > 0:
            E11, E12 = E[:, :n_nodes, :n_nodes], E[:, :n_nodes, n_nodes:]
        else:
            E11, E12 = E[:n_nodes][:, :n_nodes], E[:n_nodes][:, n_nodes:]
            

        dd_bias = jnp.concatenate([E11 - I, E12], axis=concat_dim) @ c

        if n_batch > 0 and n_A == 0:
            Ad = jnp.tile(Ad, (n_batch, 1, 1))

        return Ad, Bd, E11, E12, dd_bias, B_T

    return jax.jit(compute_dynamics_matrices)

_compute_dynamics_matrices_funcs = CompiledFunctionSet(build_compute_dynamics_matrices)


def build_compute_single_trajectory(n_nodes, n_integrate):
    """ """
    I = jnp.eye(n_nodes)
    def compute_single_trajectory(A_norm, dynamics_matrices, x0, xf, B, rho):

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
        u = (- B.T @ z[n_nodes:, :]) / (2 * rho)
        E = jnp.sum(u ** 2)

        err_costate = jnp.linalg.norm(E12 @ l0 - dd)
        err_xf = jnp.linalg.norm(x[:, -1].reshape(-1, 1) - xf)
        err = [jnp.array(err_costate, float), jnp.array(err_xf, float)]

        return E, x.T, u.T, err

    return jax.jit(compute_single_trajectory)

_compute_single_trajectory_funcs = CompiledFunctionSet(build_compute_single_trajectory)


def get_control_inputs(A_norm, x0, xf, B=None, S=None, T=1, dt=0.001, rho=1):
    """ """
    n_nodes = A_norm.shape[-1]
    n_integrate = int(numpy.round(T / dt) + 1)

    I = jnp.eye(n_nodes)
    B = I if B is None else jnp.array(B)
    S = I if S is None else jnp.array(S)

    compute_dynamics_matrices = _compute_dynamics_matrices_funcs(n_nodes, n_integrate, 0, 0)
    compute_single_trajectory = _compute_single_trajectory_funcs(n_nodes, n_integrate)

    dynamics_matrices = compute_dynamics_matrices(A_norm, S, B, T, dt, rho)
    return compute_single_trajectory(A_norm, dynamics_matrices, x0, xf, B, rho)


########################################## Block Matrices #########################################################


def build_cti_block_A_components(n_nodes, n_A, n_batch, n_integrate):
    """ """
    I_b = jnp.eye(n_nodes)
    if n_A > 1:
        T_dims, cat_dim = (0, 2, 1), 2
        I_b = jnp.tile(I_b, (n_A, 1, 1))
        expand_Ad = lambda Ad: Ad
        def split_E(E):
            return E[:, :n_nodes, :n_nodes], E[:, :n_nodes, n_nodes:]

    else:
        T_dims, cat_dim = (0, 1), 1
        expand_Ad = lambda Ad: jnp.tile(Ad, (n_batch, 1, 1))
        def split_E(E):
            return E[:n_nodes, :n_nodes], E[:n_nodes, n_nodes:]

    S_b, B_b, B_b_T = I_b, I_b, I_b.transpose(*T_dims)

    rng = jax.random.PRNGKey(0)
    # m_fake = jnp.zeros((n_A, 2*n_nodes, 2*n_nodes))
    m_fake = jax.random.normal(rng, (n_A, 2*n_nodes, 2*n_nodes))
    E11_f, E12_f = split_E(m_fake)
    Ad_f = m_fake
    dd_op_f = jnp.zeros((n_A, n_nodes, 2 * n_nodes))

    def block_A_components_func(A_norm, T = 1, rho = 1, dt = 0.001):
        """ """
        # return m_fake, E11_f, E12_f, Ad_f, S_b, B_b_T, dd_op_f

        A_norm_T = A_norm.transpose(*T_dims)
        M = jnp.concatenate([jnp.concatenate([A_norm, (-B_b @ B_b_T) / (2 * rho)], axis=cat_dim),
                             jnp.concatenate([- 2 * S_b, -A_norm_T], axis=cat_dim)],
                            axis=cat_dim - 1)


        # expm is slow in jax, can skip the double expm and just use matrix power as:
        # e^(X * T) = e^(X * dt * (T/dt)) = (e^(X * dt))^n_integrate
        Ad = jax.scipy.linalg.expm(M * dt)
        E = jax.scipy.linalg.expm(M * T)
        # E = jnp.linalg.matrix_power(Ad, n_integrate)

        Ad = expand_Ad(Ad)
        E11, E12 = split_E(E)
        dd_op = jnp.concatenate([E11 - I_b, E12], axis=cat_dim)
    
        return M, E11, E12, Ad, S_b, B_b_T, dd_op

    return jax.jit(block_A_components_func)

_CTI_block_A_component_funcs = CompiledFunctionSet(build_cti_block_A_components, [(400, 1, 20, 1001)])


def build_cti_integrate(n_integrate):
    """ """
    def cti_integrate_func(z, x0s_b, l0, Ad, Bd):
        z0 = jnp.concatenate([x0s_b, l0], axis=1).squeeze()
        z = z.at[0].set(z0)
        def body_fun(i, z):
            return z.at[i].set((Ad @ jnp.expand_dims(z[i - 1], -1) + Bd).squeeze())

        return jax.lax.fori_loop(1, n_integrate, body_fun, z)
    return jax.jit(cti_integrate_func)

_CTI_integrate_funcs = CompiledFunctionSet(build_cti_integrate, [(1001,)], make_key=lambda n: n)


def build_cti_core(n_nodes, n_batch, n_A, n_integrate):
    
    create_block_A_components = _CTI_block_A_component_funcs(n_nodes, n_A, n_batch, n_integrate)
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

        return E, x, u, err, z

    return jax.jit(cti_core)

_CTI_core_funcs = CompiledFunctionSet(build_cti_core)


def get_cti_block(A_norms, x0s, xfs, T=1, rho=1, dt = 0.001, intermediates=None):
    """ assumes multi A: TODO fixed"""
    assert x0s.shape == xfs.shape

    n_batch, n_nodes = x0s.shape 
    n_A = A_norms.shape[0] if A_norms.ndim == 3 else 1
    n_integrate = int(numpy.round(T / dt) + 1)

    CTI_core = _CTI_core_funcs(n_nodes, n_batch, n_A, n_integrate)

    return CTI_core(A_norms, x0s, xfs, T, dt, rho)


def build_compute_block_trajectory(n_nodes, n_batch, n_integrate):
    """ """
    state_tensor_shape = (n_batch, n_nodes, 1)

    big_I = jnp.eye(2 * n_nodes)

    def compute_block_trajectory(A_norms, dynamics_matrices, x0s, xfs, T, dt, rho):
        """ """
        Ad, Bd, E11, E12, dd_bias, B_T = dynamics_matrices

        x0s_b = jnp.array(x0s.reshape(state_tensor_shape))
        xfs_b = jnp.array(xfs.reshape(state_tensor_shape))

        dd = xfs_b - (E11 @ x0s_b) - dd_bias
        l0 = jnp.linalg.solve(E12, dd)

        z = jnp.zeros((n_integrate, n_batch, 2 * n_nodes))
        z0 = jnp.concatenate([x0s_b, l0], axis=1).squeeze()
        z = z.at[0].set(z0)

        # Ad = jnp.expand_dims(Ad, 0)
        # TODO: figure out why this fixes the large numerical errors
        # Ad = jnp.tile(Ad, (n_batch, 1, 1))

        def integrate_step(i, z):
            print(z[i - 1].shape, Ad.shape, Bd.shape)

            mult = Ad @ jnp.expand_dims(z[i - 1], -1)
            print(mult.shape)
            add = mult.squeeze() + Bd
            print(add.shape)
            return z.at[i].set(add)

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


def get_control_inputs_multi(A_norm, x0s, xfs, B=None, S=None, T=1, dt=0.001, rho=1):
    """ """
    n_nodes = A_norm.shape[-1]
    n_integrate = int(numpy.round(T / dt) + 1)
    n_batch = len(x0s) if x0s.ndim == 2 else 0
    n_A = A_norm.shape[0] if A_norm.ndim == 3 else 0

    I = jnp.eye(n_nodes)
    B = I if B is None else jnp.array(B)
    S = I if S is None else jnp.array(S)

    compute_dynamics_matrices = _compute_dynamics_matrices_funcs(n_nodes, n_integrate, n_batch, n_A)
    dynamics_matrices = compute_dynamics_matrices(A_norm, S, B, T, dt, rho)

    compute_block_trajectory = _compute_block_trajectory_funcs(n_nodes, n_batch, n_integrate)

    return compute_block_trajectory(A_norm, dynamics_matrices, x0s, xfs, T, dt, rho)
    # E_s, x_s, u_s, err_s = [], [], [], []
    # pbar = tqdm(total=len(x0s), desc="CTI multi jax:")
    # for x0, xf in zip(x0s, xfs):
    #     E, x, u, err = compute_single_trajectory(A_norm, dynamics_matrices, x0, xf, B, rho)
    #     E_s.append(E), x_s.append(x), u_s.append(u), err_s.append(err)
    #     pbar.update(1)

    # return numpy.array(E_s), numpy.array(x_s), numpy.array(u_s), numpy.array(err_s)

