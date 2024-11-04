import jax
import jax.numpy as jnp

import numpy

VALID_SYSTEMS = ["discrete", "continuous"]

# @jax.jit
def matrix_norm(A, c=1, system="continuous"):
    """ """
    assert system in VALID_SYSTEMS, f"Invalid system '{system}', valid sytems: {VALID_SYSTEMS}"

    # eigenvalue decomposition
    w, _ = jnp.linalg.eigh(A)
    l = jnp.abs(w).max()

    # Matrix normalization for discrete-time systems
    A_norm = A / (c + l)

    if system == 'continuous':
        # for continuous-time systems
        A_norm = A_norm - jnp.eye(A.shape[0])

    return A_norm

# @jax.jit
def get_control_inputs(A_norm, x0, xf, B=None, S=None, T=1, rho=1, n_integrate_steps=1001, device="cpu"):
    """ """


    jax.default_device(device)
    n_nodes = A_norm.shape[0]
    dt = jnp.array(T / (n_integrate_steps - 1))
    # n_integrate_steps = jnp.array(jnp.round(T / dt), int) + 1

    # jnp tensors on device

    A_norm = jnp.array(A_norm)

    x0 = jnp.array(x0.reshape(-1, 1))
    xf = jnp.array(xf.reshape(-1, 1))
    xr = jnp.zeros((n_nodes, 1))

    I = jnp.eye(n_nodes)
    B = I if B is None else jnp.array(B)
    S = I if S is None else jnp.array(S)
    T, rho, dt = jnp.array([T, rho, dt])

    M = jnp.concatenate((jnp.concatenate((A_norm, (-B @ B.T) / (2 * rho)), axis=1),
                         jnp.concatenate((-2 * S, -A_norm.T), axis=1)))

    c = jnp.concatenate([jnp.zeros((n_nodes, 1)), 2 * S @ xr], axis=0)
    c = jnp.linalg.solve(M, c)

    E = jax.scipy.linalg.expm(M * T)
    E11 = E[:n_nodes][:, :n_nodes]
    E12 = E[:n_nodes][:, n_nodes:]

    dd = xf - (E11 @ x0)  - (jnp.concatenate([E11 - I, E12], axis=1) @ c)
    l0 = jnp.linalg.solve(E12, dd)

    big_I = jnp.eye(2 * n_nodes)
    z = jnp.zeros((2 * n_nodes, n_integrate_steps))
    z = z.at[:, 0:1].set(jnp.concatenate([x0, l0], axis=0))

    Ad = jax.scipy.linalg.expm(M * dt)
    Bd = ((Ad - big_I) @ c).flatten()
    for i in jnp.arange(1, n_integrate_steps):
        z = z.at[:, i].set(Ad @ z[:, i - 1] + Bd)

    x = z[:n_nodes, :]
    u = (- B.T @ z[n_nodes:, :]) / (2 * rho)
    E = jnp.sum(u ** 2)

    err_costate = jnp.linalg.norm(E12 @ l0 - dd)
    err_xf = jnp.linalg.norm(x[:, -1].reshape(-1, 1) - xf)
    err = [jnp.array(err_costate, float), jnp.array(err_xf, float)]

    return jnp.array(E, float), numpy.array(x.T), numpy.array(u.T), err

    # return float(E), x.T.cpu().numpy(), u.T.cpu().numpy(), [float(err_costate), float(err_xf)]
