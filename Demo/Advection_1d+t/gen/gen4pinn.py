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


# Deinfe initial and boundary conditions for advection equation
# IC: f(x, 0)  = sin(pi x)
# BC: g(0, t) = sin (pi t / 2)
f = lambda x: np.sin(np.pi * x)
g = lambda t: np.sin(np.pi * t / 2)


# Advection solver
def solve_CVC(key, gp_sample, Nx, Nt, m, P):
    # Solve u_t + a(x) * u_x = 0
    # Wendroff for a(x)=V(x) - min(V(x)+ + 1.0, u(x,0)=f(x), u(0,t)=g(t)  (f(0)=g(0))
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    np.random.seed(key)
    N = gp_sample.shape[0]
    X = np.linspace(xmin, xmax, N)[:, None]
    V = lambda x: np.interp(x, X.flatten(), gp_sample)

    # Create grid
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    lam = dt / h

    # Compute advection velocity
    v_fn = lambda x: V(x) - V(x).min() + 1.0
    v = v_fn(x)

    # Initialize solution and apply initial & boundary conditions
    u = np.zeros([Nx, Nt])
    u[0, :] = g(t)
    u[:, 0] = f(x)

    # Compute finite difference operators
    a = (v[:-1] + v[1:]) / 2
    k = (1 - a * lam) / (1 + a * lam)
    K = np.eye(Nx - 1, k=0)
    K_temp = np.eye(Nx - 1, k=0)
    Trans = np.eye(Nx - 1, k=-1)

    def body_fn_x(i, carry):
        K, K_temp = carry
        K_temp = (-k[:, None]) * (Trans @ K_temp)
        K += K_temp
        return K, K_temp

    # Run loop
    for i in range(Nx - 2):
        K, K_temp = body_fn_x(i, (K, K_temp))
    # UU = u.copy()
    # K, _ = lax.fori_loop(0, Nx - 2, body_fn_x, (K, K_temp))
    D = np.diag(k) + np.eye(Nx - 1, k=-1)

    def body_fn_t(i, u):
        b = np.zeros(Nx - 1)
        b[0] = g(i * dt) - k[0] * g((i + 1) * dt)
        u[1:, i + 1] = K @ (D @ u[1:, i] + b)
        return u

    # Run loop
    for i in range(Nt - 1):
        u = body_fn_t(i, u)
    UU = u.copy()
    # UU = lax.fori_loop(0, Nt - 1, body_fn_t, u)

    # Input sensor locations and measurements
    xx = np.linspace(xmin, xmax, m)
    u = v_fn(xx)
    # Output sensor locations and measurements
    idx = np.random.randint(0, max(Nx, Nt), size=(P, 2))
    y = np.concatenate([x[idx[:, 0]][:, None], t[idx[:, 1]][:, None]], axis=1)
    s = UU[idx[:, 0], idx[:, 1]]

    return (x, t, UU), (u, y, s)


# Geneate training data corresponding to one input sample
def generate_one_training_data(key, P, Q):
    np.random.seed(key)
    # Geneate subkeys
    subkeys = np.random.randint(10000, size=(10,))
    # Generate a GP sample
    N = 512
    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = np.linspace(xmin, xmax, N)[:, None]
    K = RBF(X, X, gp_params)
    L = np.linalg.cholesky(K + jitter * np.eye(N))
    np.random.seed(subkeys[0])
    gp_sample = np.dot(L, np.random.normal(size=(N,)))
    jitter = 1e-10
    X = np.linspace(xmin, xmax, N)[:, None]
    K = RBF(X, X, gp_params)
    L = np.linalg.cholesky(K + jitter * np.eye(N))
    gp_sample = np.dot(L, np.random.normal(size=(N,)))

    v_fn = lambda x: np.interp(x, X.flatten(), gp_sample)
    u_fn = lambda x: v_fn(x) - v_fn(x).min() + 1.0

    (x, t, UU), (u, y, s) = solve_CVC(subkeys[1], gp_sample, Nx, Nt, m, P)

    x_bc1 = np.zeros((P // 2, 1))
    np.random.seed(subkeys[2])
    x_bc2 = np.random.uniform(size=(P // 2, 1))
    x_bcs = np.vstack((x_bc1, x_bc2))

    np.random.seed(subkeys[3])
    t_bc1 = np.random.uniform(size=(P // 2, 1))
    t_bc2 = np.zeros((P // 2, 1))
    t_bcs = np.vstack([t_bc1, t_bc2])

    u_train = np.tile(u, (P, 1))
    y_train = np.hstack([x_bcs, t_bcs])

    s_bc1 = g(t_bc1)
    s_bc2 = f(x_bc2)
    s_train = np.vstack([s_bc1, s_bc2])

    np.random.seed(subkeys[4])
    x_r = np.random.uniform(xmin, xmax, size=(Q, 1))
    np.random.seed(subkeys[5])
    t_r = np.random.uniform(xmin, xmax, size=(Q, 1))
    ux_r = u_fn(x_r)

    u_r_train = np.tile(u, (Q, 1))
    y_r_train = np.hstack([x_r, t_r])
    # 与验证集合不同，这里的s并非解，而是每个节点对应的输入函数值
    s_r_train = ux_r

    return u_train, y_train, s_train, u_r_train, y_r_train, s_r_train


# Geneate test data corresponding to one input sample
def generate_one_test_data(key, Nx, Nt, P):
    N = 512
    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = np.linspace(xmin, xmax, N)[:, None]
    K = RBF(X, X, gp_params)
    L = np.linalg.cholesky(K + jitter * np.eye(N))
    gp_sample = np.dot(L, np.random.normal(size=(N,)))

    (x, t, UU), (u, y, s) = solve_CVC(key, gp_sample, Nx, Nt, m, P)

    XX, TT = np.meshgrid(x, t)

    u_test = np.tile(u, (Nx * Nt, 1))
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

# Computational domain
xmin = 0.0
xmax = 1.0

tmin = 0.0
tmax = 1.0

N_train = 1000  # number of input samples
m = Nx  # number of input sensors
P_train = 200  # number of output sensors, 100 for each side
Q_train = 2000  # number of collocation points for each input sample

# Generate training data
key = np.random.permutation(np.arange(N_train))
u_bcs_train, y_bcs_train, s_bcs_train, u_res_train, y_res_train, s_res_train \
    = generate_training_data(key, N_train, P_train, Q_train)
sio.savemat('..\\data\\pinn_train.mat',
            {'u_bcs_train': u_bcs_train, 'y_bcs_train': y_bcs_train, 's_bcs_train': s_bcs_train,
             'u_res_train': u_res_train, 'y_res_train': y_res_train, 's_res_train': s_res_train})

N_valid = 50
key = np.random.permutation(np.arange(N_valid))
u_res_valid, y_res_valid, s_res_valid = generate_valid_data(key, N_valid, Nx, Nt, m)
sio.savemat('..\\data\\pinn_valid.mat',
            {'u_res_valid': u_res_valid, 'y_res_valid': y_res_valid, 's_res_valid': s_res_valid})