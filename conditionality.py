import numpy as np
import matplotlib.pyplot as plt
from autograd import hessian
import autograd.numpy as anp

def quadratic(x, y):
    return x**2 + y**2

def spherical(x, y):
    return 100 - anp.sqrt(100 - x**2 - y**2)

def rosenbrock(x, y):
    return 0.1 * (1 - x) ** 2 + 0.1 * (y - x ** 2) ** 2

def himmelblau(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

functions = {
    "Quadratic": quadratic,
    "Spherical": spherical,
    "Rosenbrock": rosenbrock,
    "Himmelblau": himmelblau
}

def wrap_func(func):
    return lambda v: func(v[0], v[1])

def compute_condition_number(func, point):
    f_wrapped = wrap_func(func)
    hess = hessian(f_wrapped)(anp.array(point))
    eigvals = np.linalg.eigvalsh(hess)
    min_abs_eig = np.min(np.abs(eigvals))
    if min_abs_eig < 1e-10:
        return np.inf
    return np.max(np.abs(eigvals)) / min_abs_eig

x_vals = np.linspace(-3, 3, 100)
y_vals = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_vals, y_vals)

fig_num = 1
for name, func in functions.items():
    Z = np.array([[func(x, y) for x in x_vals] for y in y_vals])
    cond_map = np.zeros_like(Z)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            point = (X[i, j], Y[i, j])
            try:
                cond_map[i, j] = compute_condition_number(func, point)
            except Exception:
                cond_map[i, j] = np.nan

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, np.log1p(cond_map), levels=50, cmap='inferno')
    plt.colorbar(label='log(1 + число обусловленности)')
    plt.title(f'{name} — карта обусловленности')
    plt.tight_layout()
    plt.savefig(name + "Map.png", dpi=600, bbox_inches="tight")
    plt.close()
