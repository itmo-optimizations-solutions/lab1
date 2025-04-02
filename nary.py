import numpy as np
from typing import Callable

Vector = np.ndarray
SYSTEM_EPS = np.sqrt(np.finfo(float).eps)

class NaryFunc:
    Type = Callable[[list[Vector]], float]
    func: Type
    count: int

    def __init__(self, func: Type) -> None:
        self.func = func
        self.count = 0

    def __call__(self, x: Vector) -> float:
        return self.func(*x)

    def gradient(self, x: Vector, ε: float = SYSTEM_EPS) -> Vector:
        self.count += 1
        gradient = np.zeros_like(x)
        size = len(x)
        for i in range(size):
            dx = np.zeros(size)
            h = max(ε, ε * abs(x[i]))
            dx[i] = h
            gradient[i] = (self(x + dx) - self(x - dx)) / (2 * h)
        return gradient

Scheduling = Callable[[int], float]
Rule = Callable[[NaryFunc, Vector, Vector], float]

Learning = Scheduling | Rule
