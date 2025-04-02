import scipy.optimize._linesearch as sc

from dataclasses import dataclass
from prettytable import PrettyTable
from functools import lru_cache

from nary import *
from plot import *

np.seterr(over='ignore', invalid='ignore')

def is_scheduling(algorithm) -> bool:
    return hasattr(algorithm, "__code__") and algorithm.__code__.co_argcount == 1

def gradient_descent(
    func: NaryFunc,
    start: Vector,
    learning: Learning,
    limit: float = 1e3,
    ε: float = 1e-6,
    error: float = 0.1,
) -> Tuple[Vector, int, int, list]:
    x = start.copy()
    trajectory = [x.copy()]
    k = 0

    while True:
        gradient = func.gradient(x)
        u = -gradient
        α = learning(k) if is_scheduling(learning) else learning(func, x, u)
        α = error if α is None else α  # @Nullable

        x += α * u
        trajectory.append(x.copy())

        k += 1
        if np.linalg.norm(gradient) ** 2 < ε or k > limit:
            break

    grad_count = func.count
    func.count = 0
    return x, grad_count, k, trajectory

# === Learnings

def h(k: int) -> float:
    return 1 / (k + 1) ** 0.5

def constant(λ: float) -> Scheduling:
    return lambda k: λ

def geometric() -> Scheduling:
    return lambda k: h(k) / 2 ** k

def exponential_decay(λ: float) -> Scheduling:
    return lambda k: h(k) * np.exp(-λ * k)

def polynomial_decay(α: float, β: float) -> Scheduling:
    return lambda k: h(k) * (β * k + 1) ** -α

MAX_ITER_RULE = 80

def armijo_rule(
    func: NaryFunc,
    x: Vector,
    direction: Vector,
    α: float,
    q: float,
    c: float,
) -> float | None:
    for _ in range(MAX_ITER_RULE):
        if func(x + α * direction) <= func(x) + c * α * np.linalg.norm(direction):
            return α
        α *= q
    return None

def wolfe_rule(
    func: NaryFunc,
    x: Vector,
    direction: Vector,
    α: float,
    c1: float,
    c2: float,
) -> float | None:
    for _ in range(MAX_ITER_RULE):
        if func(x + α * direction) > func(x) + c1 * α * np.dot(func.gradient(x), direction):
            α *= 0.5
        elif np.dot(func.gradient(x + α * direction), direction) < c2 * np.dot(func.gradient(x), direction):
            α *= 1.5
        else:
            return α
    return None

def armijo_rule_gen(α: float, q: float, c: float) -> Rule:
    return lambda func, x, direction: armijo_rule(func, x, direction, α=α, q=q, c=c)

def wolfe_rule_gen(α: float, c1: float, c2: float) -> Rule:
    return lambda func, x, direction: wolfe_rule(func, x, direction, α=α, c1=c1, c2=c2)

def scipy_wolfe(func: NaryFunc, x: Vector, direction: Vector) -> float:
    return sc.line_search_wolfe1(func, func.gradient, x, direction)[0]

def scipy_armijo(func: NaryFunc, x: Vector, direction: Vector) -> float:
    return sc.scalar_search_armijo(
        phi=lambda α: func(x + α * direction),
        phi0=func(x),
        derphi0=np.dot(func.gradient(x), direction),
        c1=0.4,
        alpha0=0.5,
    )[0]

# === Launcher

@dataclass
class Algorithm:
    name: str
    meta: str
    algorithm: Learning

    def get_data(self, func: NaryFunc, start: Vector) -> list:
        x, grad_count, k, _ = gradient_descent(func, start, self.algorithm)
        return [self.name] + [self.meta] + list(x) + [grad_count] + [k]

KNOWN = [
    Algorithm("Constant", "λ=0.3", constant(λ=0.3)),
    Algorithm("Constant", "λ=0.003", constant(λ=0.003)),

    Algorithm("Exponential Decay", "λ=0.01", exponential_decay(λ=0.01)),
    Algorithm("Polynomial Decay", "α=0.5, β=1", polynomial_decay(α=0.5, β=1)),

    Algorithm("Armijo", "α=0.9, q=0.5, c=0.5", armijo_rule_gen(α=0.9, q=0.5, c=0.5)),
    Algorithm("Wolfe Rule", "α=0.5, c1=1e-4, c2=0.3", wolfe_rule_gen(α=0.5, c1=1e-4, c2=0.3)),

    Algorithm("SciPy Armijo", "!", scipy_armijo),
    Algorithm("SciPy Wolfe", "!", scipy_wolfe),
]

def example_table(func: NaryFunc, start: Vector) -> PrettyTable:
    table = PrettyTable()
    table.field_names = (
        ["Method"]
        + ["Params"]
        + ["x" + str(i + 1) for i in range(len(start))]
        + ["Gradient count"]
        + ["Steps"]
    )
    table.add_rows(
        sorted(
            [algorithm.get_data(func, start) for algorithm in KNOWN],
            key=lambda x: (func(x[2:-2]), x[-1])
        )
    )
    return table

def spherical(x: float, y: float) -> float:
    return 100 - np.sqrt(100 - x ** 2 - y ** 2)

def rosenbrock(x: float, y: float) -> float:
    return 0.1 * (1 - x) ** 2 + 0.1 * (y - x ** 2) ** 2

def himmelblau(x: float, y: float) -> float:
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

def noise(x: float, y: float, amplitude: float = 0.1) -> float:
    return amplitude * (np.sin(10 * x + 20 * y) + np.cos(15 * x - 10 * y)) / 2

def noisy_function(x: float, y: float, amplitude: float, function: Callable[[float, float], float]) -> float:
    return function(x, y) + noise(x,y, amplitude)

def noisy_wrapper(x: float, y: float) -> float:
    return noisy_function(x, y, amplitude=0.1, function=spherical)

INTERESTING = \
    [
        [spherical, [-3.0, 2.0], "Quadratic function: 100 - np.sqrt(100 - x^2 - y^2)"],
        [rosenbrock, [0.0, 5.0], "Rosenbrock function: 0.1(1 - x)^2 + 0.1(y - x^2)^2"],
        [himmelblau, [1.0, 1.0], "Himmelblau function: (x^2 + y - 11)^2 + (x + y^2 - 7)^2"]
    ]

if __name__ == "__main__":
    func = NaryFunc(noisy_wrapper)
    start = np.array([0.0, 5.0])
    print(example_table(func, start))
    x, _, _, trajectory = gradient_descent(func, start, armijo_rule_gen(α=0.9, q=0.5, c=0.5))
    plot_gradient(lambda x: func(x), len(start) == 1, len(start) == 2, trajectory, name="Noisy Rosenbroke Function")
    print(x)
