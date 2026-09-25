"""Weighted matrix decision diagrams with epsilon-approximate node sharing.

Quadrant order is (M00, M01, M10, M11); the root has var=n-1.
Approximation is local to normalized node edges and has no global error bound.
"""

import json
import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class DDNode:
    var: int
    edges: tuple["DDEdge", ...] = ()


@dataclass(frozen=True, eq=False)
class DDEdge:
    weight: complex
    node: DDNode


TERMINAL = DDNode(-1)


def quantize_complex(c: complex, epsilon: float) -> tuple[int, int]:
    """Round each component onto the epsilon grid (ties to even)."""
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    value = complex(c)
    if not np.isfinite(value.real) or not np.isfinite(value.imag):
        raise ValueError("edge weights must be finite")
    return (int(np.rint(value.real / epsilon)),
            int(np.rint(value.imag / epsilon)))


class UniqueTable:
    """Intern normalized nodes by level, child identity and quantized weights."""

    def __init__(self, epsilon: float = 1e-4):
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.epsilon = float(epsilon)
        self._nodes: dict[tuple, DDNode] = {}

    def lookup_or_create(self, var: int, edges: tuple[DDEdge, ...]) -> DDEdge:
        if not isinstance(var, int) or var < 0:
            raise ValueError("var must be a nonnegative integer")
        edges = tuple(edges)
        if len(edges) != 4 or any(not isinstance(edge, DDEdge) for edge in edges):
            raise ValueError("exactly four DDEdge children are required")
        for edge in edges:
            if edge.node.var != var - 1 and not (edge.weight == 0 and edge.node is TERMINAL):
                raise ValueError("each child must occupy the immediately lower level")
            if not np.isfinite(edge.weight.real) or not np.isfinite(edge.weight.imag):
                raise ValueError("edge weights must be finite")

        pivot = next((edge.weight for edge in edges
                      if abs(edge.weight) > self.epsilon), None)
        if pivot is None:
            return DDEdge(0j, TERMINAL)

        normalized = tuple(DDEdge(complex(edge.weight / pivot), edge.node)
                           if abs(edge.weight) > self.epsilon
                           else DDEdge(0j, TERMINAL) for edge in edges)
        key = (var, tuple((id(edge.node), quantize_complex(edge.weight, self.epsilon))
                          for edge in normalized))
        node = self._nodes.get(key)
        if node is None:
            node = DDNode(var, normalized)
            self._nodes[key] = node
        return DDEdge(complex(pivot), node)

    def get_active_node_count(self) -> int:
        """Number of interned nonterminal nodes retained by this table."""
        return len(self._nodes)


def matrix_to_dd(matrix: np.ndarray, unique_table: UniqueTable) -> DDEdge:
    """Build a diagram for a square power-of-two complex matrix."""
    if not isinstance(unique_table, UniqueTable):
        raise TypeError("unique_table must be a UniqueTable")
    data = np.asarray(matrix, dtype=np.complex128)
    if data.ndim != 2 or data.shape[0] != data.shape[1] or data.shape[0] == 0:
        raise ValueError("matrix must be nonempty and square")
    size = data.shape[0]
    if size & (size - 1):
        raise ValueError("matrix dimension must be a power of two")
    if not np.all(np.isfinite(data)):
        raise ValueError("matrix entries must be finite")

    def build(block: np.ndarray, var: int) -> DDEdge:
        if var < 0:
            return DDEdge(complex(block[0, 0]), TERMINAL)
        half = block.shape[0] // 2
        children = (build(block[:half, :half], var - 1),
                    build(block[:half, half:], var - 1),
                    build(block[half:, :half], var - 1),
                    build(block[half:, half:], var - 1))
        return unique_table.lookup_or_create(var, children)

    return build(data, size.bit_length() - 2)


def dd_to_matrix(edge: DDEdge, n_qubits: int) -> np.ndarray:
    """Expand an edge to a dense 2**n by 2**n complex matrix."""
    if not isinstance(edge, DDEdge) or not isinstance(n_qubits, int) or n_qubits < 0:
        raise ValueError("provide a DDEdge and a nonnegative qubit count")
    if edge.node is not TERMINAL and edge.node.var != n_qubits - 1:
        raise ValueError("root node level does not match n_qubits")

    def expand(current: DDEdge, level: int) -> np.ndarray:
        if current.weight == 0:
            return np.zeros((1 << level, 1 << level), dtype=np.complex128)
        if level == 0:
            if current.node is not TERMINAL:
                raise ValueError("nonterminal node at scalar level")
            return np.array([[current.weight]], dtype=np.complex128)
        if current.node is TERMINAL:
            raise ValueError("nonzero terminal edge above scalar level")
        if current.node.var != level - 1 or len(current.node.edges) != 4:
            raise ValueError("malformed decision diagram")
        a, b, c, d = (expand(child, level - 1) for child in current.node.edges)
        return current.weight * np.block([[a, b], [c, d]])

    return expand(edge, n_qubits)


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    raw = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    random_density = raw @ raw.conj().T
    random_density /= np.trace(random_density)
    density = (1 - 1e-7) * np.eye(16, dtype=np.complex128) / 16
    density += 1e-7 * random_density

    exact_table = UniqueTable(epsilon=1e-14)
    approximate_table = UniqueTable(epsilon=1e-4)
    exact_edge = matrix_to_dd(density, exact_table)
    approximate_edge = matrix_to_dd(density, approximate_table)
    exact_error = float(np.max(np.abs(dd_to_matrix(exact_edge, 4) - density)))
    approximate_error = float(np.max(np.abs(dd_to_matrix(approximate_edge, 4) - density)))
    exact_count = exact_table.get_active_node_count()
    approximate_count = approximate_table.get_active_node_count()
    assert exact_error < 1e-12, exact_error
    assert approximate_error < 1e-6, approximate_error
    assert approximate_count < exact_count, (approximate_count, exact_count)
    print(json.dumps({"exact_nodes": exact_count,
                      "approximate_nodes": approximate_count,
                      "exact_max_error": exact_error,
                      "approximate_max_error": approximate_error}, indent=2))
