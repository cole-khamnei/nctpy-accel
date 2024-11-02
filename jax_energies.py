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


def make_block_cti_A_components_function(n_nodes, n_A):
    """ """
    I_b = jnp.eye(n_nodes)
    if n_A > 1:
        T_dims, cat_dim = (0, 2, 1), 2
        I_b = jnp.tile(I_b, (n_A, 1, 1))
    else:
        T_dims, cat_dim = (0, 1), 1

    def block_A_components(A_norm, T = 1, rho = 1, dt = 0.001):
        """ """
        S_b, B_b, B_b_T = I_b, I_b, I_b.transpose(*T_dims)
        A_norm_T = A_norm.transpose(*T_dims)
        M = jnp.concatenate([jnp.concatenate([A_norm, (-B_b @ B_b_T) / (2 * rho)], axis=cat_dim),
                             jnp.concatenate([- 2 * S_b, -A_norm_T], axis=cat_dim)],
                            axis=cat_dim - 1)
        E = jax.scipy.linalg.expm(M * T)
        Ad = jax.scipy.linalg.expm(M * dt)
        E11, E12 = E[:, :n_nodes][:, :, :n_nodes], E[:, :n_nodes][:, :, n_nodes:]
        dd_op = jnp.concatenate([E11 - I_b, E12], axis=cat_dim)
        return M, E11, E12, Ad, S_b, B_b_T, dd_op

    # return block_A_components
    return jax.jit(block_A_components)


CTI_INTERMEDIATE_FUNCTIONS = {"400_1": make_block_cti_A_components_function(400, 0)}

def get_cti_A_components(A_norm, T=1, rho=1, dt = 0.001):
    """ """
    A_norm = jnp.array(A_norm)
    n_nodes = A_norm.shape[-1]
    func_tag = f"{n_nodes}_{A_norm.shape[0]}" if len(A_norm.shape) == 3 else f"{n_nodes}_1"

    if func_tag in CTI_INTERMEDIATE_FUNCTIONS:
        get_cti_A_components_func = CTI_INTERMEDIATE_FUNCTIONS[func_tag]
    else:
        get_cti_A_components_func = make_block_cti_A_components_function(*[int(s) for s in func_tag.split("_")])
        CTI_INTERMEDIATE_FUNCTIONS[func_tag] = get_cti_A_components_func  

    return get_cti_A_components_func(A_norm, T=T, rho=rho, dt=dt)


def make_cti_integrate_function(n_integrate_steps):
    """ """
    def cti_integrate_func(z, x0s_b, l0, Ad, Bd):
    # z = jnp.zeros((n_batch, 2 * n_nodes, n_integrate_steps))
        z = z.at[:, :, :1].set(jnp.concatenate([x0s_b, l0], axis=1))
        for i in range(1, n_integrate_steps):
            z = z.at[:, :, i:i+1].set(Ad @ z[:, :, i-1:i] + Bd)
        return z

    return jax.jit(cti_integrate_func)


class CTIIntegrateFunctions():
    def __init__(self, n_integrate_steps_list=[1001]):
        self.make_function = make_cti_integrate_function
        self.functions = {n: self.make_function(n) for n in n_integrate_steps_list}

    def __getitem__(self, index):
        if index not in self.functions:
            self.functions[index] = self.make_function(index)
        return self.functions[index]
            
CTI_INTEGRATE_FUNCTIONS = CTIIntegrateFunctions()

def cti_integrate(z, x0s_b, l0, Ad, Bd, n_integrate_steps):
    """ """
    return CTI_INTEGRATE_FUNCTIONS[n_integrate_steps](z, x0s_b, l0, Ad, Bd)

def get_cti_block(A_norms, x0s, xfs, T=1, rho=1, dt = 0.001, intermediates=None):
    """ """
    if intermediates is None:
        assert len(A_norms) == len(x0s) or (len(A_norms.shape) == 2)
        with Timer("get CTI A components") as t:
            M, E11, E12, Ad, S, B_T, dd_op = get_cti_A_components(A_norms, T=T, rho=rho, dt=dt)
    else:
        M, E11, E12, Ad, S, B_T, dd_op = intermediates

    x0s_b = jnp.array(x0s.reshape(*x0s.shape, 1))
    xfs_b = jnp.array(xfs.reshape(*xfs.shape, 1))
    xrs_b = jnp.zeros(x0s_b.shape)
    zero_array = jnp.zeros(xrs_b.shape)

    n_batch, n_nodes = x0s.shape
    n_integrate_steps = int(jnp.round(T / dt)) + 1
    
    with Timer("intermediates") as t:
        c = jnp.concatenate([zero_array, 2 * S @ xrs_b], axis=1)
        c = jnp.linalg.solve(M, c)
        
        dd = xfs_b - (E11 @ x0s_b) - (dd_op @ c)
        l0 = jnp.linalg.solve(E12, dd)

        # Make Big matrices
        big_I = jnp.eye(2 * n_nodes)
        Bd = (Ad - big_I) @ c

    with Timer("integrate") as t:
        # Integrate trajectory
        z = jnp.zeros((n_batch, 2 * n_nodes, n_integrate_steps))
        z = CTI_INTEGRATE_FUNCTIONS[n_integrate_steps](z, x0s_b, l0, Ad, Bd)
        
    with Timer("final outputs") as t:
        x = z[:, :n_nodes, :]
        u = (- B_T @ z[:, n_nodes:, :]) / (2 * rho)
        E = jnp.trapezoid(u ** 2, axis=1).sum(axis=1)

        err_xf = jnp.linalg.norm(x[:, :, -1:] - xfs_b, axis=1)
        err_costate = jnp.linalg.norm(E12 @ l0 - dd, axis=1)
        err = jnp.concatenate([err_xf, err_costate], axis=1)
        
    return jnp.array(E), jnp.array(x.transpose(0, 2, 1)), jnp.array(u.transpose(0, 2, 1)), jnp.array(err)
