# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/2/4 19:02
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：gen4pinn.py
@File ：gen4pinn.py
"""

import numpy as np
import scipy.io as sio

# Use double precision to generate data (due to GP sampling)
def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs ** 2, axis=2)
    return output_scale * np.exp(-0.5 * r2)


# A diffusion-reaction numerical solver
def solve_ADR(key, Nx, Nt, P, length_scale):
    """Solve 1D
    u_t = (k(x) u_x)_x - v(x) u_x + g(u) + f(x)
    with zero initial and boundary conditions.
    """
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    k = lambda x: 0.01 * np.ones_like(x)
    v = lambda x: np.zeros_like(x)
    g = lambda u: 0.01 * u ** 2
    dg = lambda u: 0.02 * u
    u0 = lambda x: np.zeros_like(x)

    # Generate a GP sample
    N = 512
    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = np.linspace(xmin, xmax, N)[:, None]
    K = RBF(X, X, gp_params)
    L = np.linalg.cholesky(K + jitter * np.eye(N))
    np.random.seed(key)
    gp_sample = np.dot(L, np.random.normal(size=(N,)))
    # Create a callable interpolation function
    f_fn = lambda x: np.interp(x, X.flatten(), gp_sample)

    # Create grid
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h ** 2

    # Compute coefficients and forcing
    k = k(x)
    v = v(x)
    f = f_fn(x)

    # Compute finite difference operators
    D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)
    D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
    D3 = np.eye(Nx - 2)
    M = -np.diag(D1 @ k) @ D1 - 4 * np.diag(k) @ D2
    m_bond = 8 * h2 / dt * D3 + M[1:-1, 1:-1]
    v_bond = 2 * h * np.diag(v[1:-1]) @ D1[1:-1, 1:-1] + 2 * h * np.diag(
        v[2:] - v[: Nx - 2]
    )
    mv_bond = m_bond + v_bond
    c = 8 * h2 / dt * D3 - M[1:-1, 1:-1] - v_bond

    # Initialize solution and apply initial condition
    u = np.zeros((Nx, Nt))
    u[:, 0] = u0(x)

    # Time-stepping update
    def body_fn(i, u):
        gi = g(u[1:-1, i])
        dgi = dg(u[1:-1, i])
        h2dgi = np.diag(4 * h2 * dgi)
        A = mv_bond - h2dgi
        b1 = 8 * h2 * (0.5 * f[1:-1] + 0.5 * f[1:-1] + gi)
        b2 = (c - h2dgi) @ u[1:-1, i].T
        u[1:-1, i + 1] = np.linalg.solve(A, b1 + b2)
        return u

    # Run loop
    for i in range(Nt - 1):
        u = body_fn(i, u)
    UU = u.copy()

    # Input sensor locations and measurements
    xx = np.linspace(xmin, xmax, m)
    u = f_fn(xx)
    # Output sensor locations and measurements
    np.random.seed(key * 2)
    idx = np.random.randint(0, max(Nx, Nt), size=(P, 2))
    y = np.concatenate([x[idx[:, 0]][:, None], t[idx[:, 1]][:, None]], axis=1)
    s = UU[idx[:, 0], idx[:, 1]]
    # x, t: sampled points on grid
    return (x, t, UU), (u, y, s)


# Geneate training data corresponding to one input sample
def generate_one_training_data(key, P, Q):
    np.random.seed(key)
    # Numerical solution
    (x, t, UU), (u, y, s) = solve_ADR(key, Nx, Nt, P, length_scale)

    # Geneate subkeys
    subkeys = np.random.randint(2022, size=(4,))

    # Sample points from the boundary and the inital conditions
    # Here we regard the initial condition as a special type of boundary conditions
    x_bc1 = np.zeros((P // 3, 1))
    x_bc2 = np.ones((P // 3, 1))
    np.random.seed(subkeys[0])
    x_bc3 = np.random.uniform(size=(P // 3, 1))
    x_bcs = np.vstack((x_bc1, x_bc2, x_bc3))
    np.random.seed(subkeys[1])
    t_bc1 = np.random.uniform(size=(P // 3 * 2, 1))
    t_bc2 = np.zeros((P // 3, 1))
    t_bcs = np.vstack([t_bc1, t_bc2])

    # Training data for BC and IC
    u_train = np.tile(u, (P, 1))
    y_train = np.hstack([x_bcs, t_bcs])
    s_train = np.zeros((P, 1))

    # Sample collocation points
    np.random.seed(subkeys[2])
    x_r_idx = np.random.choice(np.arange(Nx), size=(Q, 1))
    x_r = x[x_r_idx]
    np.random.seed(subkeys[3])
    t_r = np.random.uniform(size=(Q, 1))

    # Training data for the PDE residual

    u_r_train = np.tile(u, (Q, 1))
    y_r_train = np.hstack([x_r, t_r])
    # 与验证集合不同，这里的s并非解，而是每个节点对应的输入函数值
    s_r_train = u[x_r_idx]

    return u_train, y_train, s_train, u_r_train, y_r_train, s_r_train


# Geneate test data corresponding to one input sample
def generate_one_test_data(key, Nx, Nt, P):

    (x, t, UU), (u, y, s) = solve_ADR(key, Nx, Nt, P, length_scale)

    XX, TT = np.meshgrid(x, t)

    u_test = np.tile(u, (P ** 2, 1))
    y_test = np.hstack([XX.flatten()[:, None], TT.flatten()[:, None]])
    s_test = UU.T.flatten()

    return u_test, y_test, s_test


# Geneate training data corresponding to N input sample
def generate_training_data(keys, N, P, Q):
    u_list = []
    y_list = []
    s_list = []
    u_r_list = []
    y_r_list = []
    s_r_list = []
    k = 0
    for key in keys:
        u_train, y_train, s_train, u_r_train, y_r_train, s_r_train = generate_one_training_data(key, P, Q)
        k += 1
        print(k)
        u_list.append(u_train)
        y_list.append(y_train)
        s_list.append(s_train)
        u_r_list.append(u_r_train)
        y_r_list.append(y_r_train)
        s_r_list.append(s_r_train)

    u_train = np.array(u_list, dtype=np.float32)
    y_train = np.array(y_list, dtype=np.float32)
    s_train = np.array(s_list, dtype=np.float32)

    u_r_train = np.array(u_r_list, dtype=np.float32)
    y_r_train = np.array(y_r_list, dtype=np.float32)
    s_r_train = np.array(s_r_list, dtype=np.float32)

    return u_train, y_train, s_train, u_r_train, y_r_train, s_r_train


# Geneate test data corresponding to N input sample
def generate_valid_data(keys, N, Nx, Nt, P):
    u_list = []
    y_list = []
    s_list = []
    for key in keys:
        u_r, y_r, s_r = generate_one_test_data(key, Nx, Nt, P)
        u_list.append(u_r)
        y_list.append(y_r)
        s_list.append(s_r)

    u_test = np.array(u_list, dtype=np.float32)
    y_test = np.array(y_list, dtype=np.float32)
    s_test = np.array(s_list, dtype=np.float32)

    return u_test, y_test, s_test


# GRF length scale
length_scale = 0.2

# Resolution of the solution
Nx = 100
Nt = 100

N_train = 5000  # number of input samples
m = Nx  # number of input sensors
P_train = 300  # number of output sensors, 100 for each side
Q_train = 200  # number of collocation points for each input sample

key = np.random.permutation(np.arange(N_train))
u_bcs_train, y_bcs_train, s_bcs_train, u_res_train, y_res_train, s_res_train = generate_training_data(key, N_train, P_train, Q_train)
sio.savemat('..\\data\\pinn_train.mat', {'u_bcs_train': u_bcs_train, 'y_bcs_train': y_bcs_train, 's_bcs_train': s_bcs_train,
                                'u_res_train': u_res_train, 'y_res_train': y_res_train, 's_res_train': s_res_train})

N_valid = 200
key = np.random.permutation(np.arange(N_train, N_train + N_valid))
u_res_valid, y_res_valid, s_res_valid = generate_valid_data(key, N_valid, Nx, Nt, m)
sio.savemat('..\\data\\pinn_valid.mat', {'u_res_valid': u_res_valid, 'y_res_valid': y_res_valid, 's_res_valid': s_res_valid})
