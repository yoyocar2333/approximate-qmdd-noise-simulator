"""Memoized algebra on weighted matrix decision diagrams.

The optional tables let a circuit reuse interning and computation across gates.
NumPy is used only in the executable reference test at the bottom of this file.
"""

import json
import os

import numpy as np

from dd_core import DDEdge, TERMINAL, UniqueTable, dd_to_matrix, matrix_to_dd, quantize_complex


class ComputeTable:
    """Cache operation results using node identity and quantized edge weights."""

    def __init__(self, epsilon: float = 1e-4):
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.epsilon = float(epsilon)
        self._cache: dict[tuple, DDEdge] = {}

    def key(self, op_type: str, edge_a: DDEdge, edge_b: DDEdge | None,
            level: int) -> tuple:
        return (op_type, level, id(edge_a.node),
                id(edge_b.node) if edge_b is not None else None,
                quantize_complex(edge_a.weight, self.epsilon),
                quantize_complex(edge_b.weight, self.epsilon) if edge_b is not None else None)

    def get(self, key: tuple) -> DDEdge | None:
        return self._cache.get(key)

    def put(self, key: tuple, result: DDEdge) -> DDEdge:
        self._cache[key] = result
        return result


def _tables(unique_table: UniqueTable | None,
            compute_table: ComputeTable | None) -> tuple[UniqueTable, ComputeTable]:
    table = unique_table if unique_table is not None else UniqueTable()
    cache = compute_table if compute_table is not None else ComputeTable(table.epsilon)
    if not isinstance(table, UniqueTable) or not isinstance(cache, ComputeTable):
        raise TypeError("expected UniqueTable and ComputeTable")
    if cache.epsilon != table.epsilon:
        raise ValueError("compute and unique table epsilon must match")
    return table, cache


def _check(edge: DDEdge, level: int) -> None:
    if not isinstance(edge, DDEdge) or not isinstance(level, int) or level < 0:
        raise ValueError("invalid edge or qubit count")
    if edge.weight != 0 and edge.node.var != level - 1:
        raise ValueError("edge level does not match qubit count")


def _child(edge: DDEdge, index: int, level: int) -> DDEdge:
    if edge.weight == 0:
        return DDEdge(0j, TERMINAL)
    if edge.node.var != level - 1 or len(edge.node.edges) != 4:
        raise ValueError("malformed DD node")
    part = edge.node.edges[index]
    return DDEdge(edge.weight * part.weight, part.node)


def _add(a: DDEdge, b: DDEdge, level: int,
         table: UniqueTable, cache: ComputeTable) -> DDEdge:
    if a.weight == 0:
        return b
    if b.weight == 0:
        return a
    key = cache.key("add", a, b, level)
    hit = cache.get(key)
    if hit is not None:
        return hit
    if level == 0:
        result = DDEdge(a.weight + b.weight, TERMINAL)
    else:
        result = table.lookup_or_create(level - 1, tuple(
            _add(_child(a, i, level), _child(b, i, level), level - 1, table, cache)
            for i in range(4)))
    return cache.put(key, result)


def _mult(a: DDEdge, b: DDEdge, level: int,
          table: UniqueTable, cache: ComputeTable) -> DDEdge:
    if a.weight == 0 or b.weight == 0:
        return DDEdge(0j, TERMINAL)
    key = cache.key("mult", a, b, level)
    hit = cache.get(key)
    if hit is not None:
        return hit
    if level == 0:
        result = DDEdge(a.weight * b.weight, TERMINAL)
    else:
        parts = []
        for row in range(2):
            for col in range(2):
                left = _mult(_child(a, row * 2, level),
                             _child(b, col, level), level - 1, table, cache)
                right = _mult(_child(a, row * 2 + 1, level),
                              _child(b, col + 2, level), level - 1, table, cache)
                parts.append(_add(left, right, level - 1, table, cache))
        result = table.lookup_or_create(level - 1, tuple(parts))
    return cache.put(key, result)


def _dagger(edge: DDEdge, level: int,
            table: UniqueTable, cache: ComputeTable) -> DDEdge:
    if edge.weight == 0:
        return DDEdge(0j, TERMINAL)
    key = cache.key("dagger", edge, None, level)
    hit = cache.get(key)
    if hit is not None:
        return hit
    if level == 0:
        result = DDEdge(edge.weight.conjugate(), TERMINAL)
    else:
        parts = tuple(_dagger(_child(edge, index, level), level - 1, table, cache)
                      for index in (0, 2, 1, 3))
        result = table.lookup_or_create(level - 1, parts)
    return cache.put(key, result)


def dd_add(edge1: DDEdge, edge2: DDEdge, n_qubits: int,
           unique_table: UniqueTable | None = None,
           compute_table: ComputeTable | None = None) -> DDEdge:
    table, cache = _tables(unique_table, compute_table)
    _check(edge1, n_qubits)
    _check(edge2, n_qubits)
    return _add(edge1, edge2, n_qubits, table, cache)


def dd_mult(edge1: DDEdge, edge2: DDEdge, n_qubits: int,
            unique_table: UniqueTable | None = None,
            compute_table: ComputeTable | None = None) -> DDEdge:
    table, cache = _tables(unique_table, compute_table)
    _check(edge1, n_qubits)
    _check(edge2, n_qubits)
    return _mult(edge1, edge2, n_qubits, table, cache)


def dd_dagger(edge: DDEdge, unique_table: UniqueTable | None = None,
              compute_table: ComputeTable | None = None) -> DDEdge:
    table, cache = _tables(unique_table, compute_table)
    level = edge.node.var + 1 if edge.weight != 0 else 0
    _check(edge, level)
    return _dagger(edge, level, table, cache)


def dd_kron(edge_top: DDEdge, edge_bottom: DDEdge,
            unique_table: UniqueTable | None = None,
            compute_table: ComputeTable | None = None,
            top_qubits: int | None = None,
            bottom_qubits: int | None = None) -> DDEdge:
    table, cache = _tables(unique_table, compute_table)
    top = edge_top.node.var + 1 if top_qubits is None else top_qubits
    bottom = edge_bottom.node.var + 1 if bottom_qubits is None else bottom_qubits
    if top < 0 or bottom < 0:
        raise ValueError("explicit qubit counts required for zero edges")
    _check(edge_top, top)
    _check(edge_bottom, bottom)

    def attach(edge: DDEdge, level: int) -> DDEdge:
        if edge.weight == 0 or edge_bottom.weight == 0:
            return DDEdge(0j, TERMINAL)
        key = cache.key("kron:" + str(bottom), edge, edge_bottom, level)
        hit = cache.get(key)
        if hit is not None:
            return hit
        if level == 0:
            result = DDEdge(edge.weight * edge_bottom.weight, edge_bottom.node)
        else:
            result = table.lookup_or_create(level + bottom - 1, tuple(
                attach(_child(edge, i, level), level - 1) for i in range(4)))
        return cache.put(key, result)

    return attach(edge_top, top)


def apply_unitary(rho_edge: DDEdge, u_edge: DDEdge, n_qubits: int,
                  unique_table: UniqueTable | None = None,
                  compute_table: ComputeTable | None = None) -> DDEdge:
    table, cache = _tables(unique_table, compute_table)
    return dd_mult(dd_mult(u_edge, rho_edge, n_qubits, table, cache),
                   dd_dagger(u_edge, table, cache), n_qubits, table, cache)


def apply_kraus_channel(rho_edge: DDEdge, kraus_edges: list[DDEdge],
                        n_qubits: int, unique_table: UniqueTable | None = None,
                        compute_table: ComputeTable | None = None) -> DDEdge:
    table, cache = _tables(unique_table, compute_table)
    _check(rho_edge, n_qubits)
    if not kraus_edges:
        raise ValueError("at least one Kraus operator is required")
    total = DDEdge(0j, TERMINAL)
    for operator in kraus_edges:
        _check(operator, n_qubits)
        term = dd_mult(dd_mult(operator, rho_edge, n_qubits, table, cache),
                       dd_dagger(operator, table, cache), n_qubits, table, cache)
        total = dd_add(total, term, n_qubits, table, cache)
    return total


if __name__ == "__main__":
    table = UniqueTable(1e-10)
    cache = ComputeTable(table.epsilon)
    identity = np.eye(2, dtype=np.complex128)
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    z = np.diag([1, -1]).astype(np.complex128)
    h = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    h_full = np.kron(np.kron(h, identity), identity)
    cnot = np.kron(np.diag([1, 0]), np.eye(4)) + np.kron(
        np.diag([0, 1]), np.kron(x, identity))
    basis = np.zeros(8, dtype=np.complex128)
    basis[0] = 1
    reference = np.outer(basis, basis.conj())
    state = matrix_to_dd(reference, table)
    h_edge = dd_kron(matrix_to_dd(h, table), matrix_to_dd(np.eye(4), table),
                     table, cache)
    cnot_edge = matrix_to_dd(cnot, table)
    state = apply_unitary(state, h_edge, 3, table, cache)
    reference = h_full @ reference @ h_full.conj().T
    h_error = float(np.max(np.abs(dd_to_matrix(state, 3) - reference)))
    state = apply_unitary(state, cnot_edge, 3, table, cache)
    reference = cnot @ reference @ cnot.conj().T
    cnot_error = float(np.max(np.abs(dd_to_matrix(state, 3) - reference)))
    p = 0.12
    paulis = (identity, x, y, z)
    coefficients = (np.sqrt(1 - p),) + (np.sqrt(p / 3),) * 3
    operators = [np.kron(np.kron(identity, identity), coefficient * pauli)
                 for coefficient, pauli in zip(coefficients, paulis)]
    channel_edges = [matrix_to_dd(operator, table) for operator in operators]
    state = apply_kraus_channel(state, channel_edges, 3, table, cache)
    reference = sum((operator @ reference @ operator.conj().T
                     for operator in operators), np.zeros((8, 8), dtype=np.complex128))
    error = float(np.max(np.abs(dd_to_matrix(state, 3) - reference)))
    assert max(h_error, cnot_error, error) < 1e-8, (h_error, cnot_error, error)
    print(json.dumps({"h_error": h_error, "cnot_error": cnot_error,
                      "channel_error": error, "active_nodes": table.get_active_node_count(),
                      "cached_operations": len(cache._cache)}, indent=2))
