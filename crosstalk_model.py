"""Instruction-ordered, topology-aware noise simulation on matrix DDs.

Physical qubit q is bit q of a Qiskit basis index. The DD root is q=n-1.
For each one/two-qubit gate: ideal gate, depolarizing channel, then one
exp(-i*theta*ZZ) per undirected hardware edge incident to an active qubit.
Active-active edges (including a two-qubit gate's own edge) are included once.
This is a phenomenological per-instruction model, not pulse scheduling.

Only local 2x2/4x4 operators are dense; initialization, embedding, evolution
and diagnostics use DDs. Dense output is an explicit caller-side operation.
"""

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from numbers import Integral
from types import MappingProxyType

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap

from dd_core import DDEdge, TERMINAL, UniqueTable, matrix_to_dd
from dd_ops import ComputeTable, apply_kraus_channel, apply_unitary


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _probability(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


@dataclass(frozen=True)
class HardwareTopology:
    """Immutable undirected hardware adjacency and per-gate noise strengths.

    p is the replacement-strength parameter of D_p(rho)=(1-p)rho+p I/d
    on the gate subsystem, not the total nonidentity Pauli probability.
    Circuit qubit indices are physical indices; supply an already mapped
    circuit with the same width as the topology, including idle spectators.
    """

    n_qubits: int
    coupling_map: Mapping | Iterable = field(default_factory=tuple)
    p_depol: float = 0.001
    p_cx_depol: float = 0.01
    zz_theta: float = 0.01

    def __post_init__(self) -> None:
        n = _integer(self.n_qubits, "n_qubits", 1)
        adjacency = {q: set() for q in range(n)}
        source = self.coupling_map
        if isinstance(source, CouplingMap):
            if source.size() != n:
                raise ValueError("CouplingMap width does not match n_qubits")
            pairs = source.get_edges()
        elif isinstance(source, Mapping):
            pairs = []
            for q, neighbors in source.items():
                q = _integer(q, "adjacency key")
                if q >= n:
                    raise ValueError("adjacency key outside hardware")
                pairs.extend((q, neighbor) for neighbor in neighbors)
        else:
            pairs = source
        for pair in pairs:
            pair = tuple(pair)
            if len(pair) != 2:
                raise ValueError("each coupling must contain two qubit indices")
            a, b = (_integer(q, "coupling qubit") for q in pair)
            if a >= n or b >= n or a == b:
                raise ValueError("couplings must connect distinct hardware qubits")
            adjacency[a].add(b)
            adjacency[b].add(a)
        theta = float(self.zz_theta)
        if not np.isfinite(theta):
            raise ValueError("zz_theta must be finite")
        object.__setattr__(self, "n_qubits", n)
        object.__setattr__(self, "p_depol", _probability(self.p_depol, "p_depol"))
        object.__setattr__(self, "p_cx_depol", _probability(self.p_cx_depol, "p_cx_depol"))
        object.__setattr__(self, "zz_theta", theta)
        object.__setattr__(self, "coupling_map", MappingProxyType(
            {q: frozenset(neighbors) for q, neighbors in adjacency.items()}))

    def neighbors(self, qubit: int) -> tuple:
        qubit = _integer(qubit, "qubit")
        if qubit >= self.n_qubits:
            raise ValueError("qubit outside hardware")
        return tuple(sorted(self.coupling_map[qubit]))

    @property
    def edges(self) -> tuple:
        return tuple((q, neighbor) for q in range(self.n_qubits)
                     for neighbor in self.neighbors(q) if q < neighbor)

    @classmethod
    def linear(cls, n_qubits: int, **parameters) -> "HardwareTopology":
        n = _integer(n_qubits, "n_qubits", 1)
        return cls(n, [(q, q + 1) for q in range(n - 1)], **parameters)

    @classmethod
    def ring(cls, n_qubits: int, **parameters) -> "HardwareTopology":
        n = _integer(n_qubits, "n_qubits", 3)
        return cls(n, [(q, (q + 1) % n) for q in range(n)], **parameters)

    @classmethod
    def heavy_hex(cls, distance: int = 3, **parameters) -> "HardwareTopology":
        distance = _integer(distance, "distance", 1)
        if distance % 2 == 0:
            raise ValueError("heavy-hex distance must be odd")
        return cls.from_coupling_map(CouplingMap.from_heavy_hex(distance), **parameters)

    @classmethod
    def from_coupling_map(cls, coupling_map: CouplingMap,
                          **parameters) -> "HardwareTopology":
        if not isinstance(coupling_map, CouplingMap):
            raise TypeError("coupling_map must be a Qiskit CouplingMap")
        return cls(coupling_map.size(), coupling_map, **parameters)

    def to_dict(self) -> dict:
        return {"schema_version": 1, "n_qubits": self.n_qubits,
                "coupling_map": {str(q): list(self.neighbors(q))
                                 for q in range(self.n_qubits)},
                "p_depol": self.p_depol, "p_cx_depol": self.p_cx_depol,
                "zz_theta": self.zz_theta}

    def to_json(self, path: str | os.PathLike) -> None:
        path = os.path.abspath(os.fspath(path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2, allow_nan=False)
            stream.write("\n")

    @classmethod
    def from_json(cls, path: str | os.PathLike) -> "HardwareTopology":
        with open(os.fspath(path), encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict) or data.get("schema_version", 1) != 1:
            raise ValueError("unsupported hardware JSON schema")
        allowed = {"schema_version", "n_qubits", "coupling_map",
                   "p_depol", "p_cx_depol", "zz_theta"}
        if set(data) - allowed or "n_qubits" not in data or "coupling_map" not in data:
            raise ValueError("hardware JSON has missing or unknown fields")
        data.pop("schema_version", None)
        if isinstance(data["coupling_map"], dict):
            data["coupling_map"] = {int(q): neighbors
                                    for q, neighbors in data["coupling_map"].items()}
        return cls(**data)


def get_zz_crosstalk_unitary(theta: float) -> np.ndarray:
    """Return exp(-i*theta*Z tensor Z); Qiskit RZZ uses angle 2*theta."""
    theta = float(theta)
    if not np.isfinite(theta):
        raise ValueError("theta must be finite")
    return np.diag(np.exp(-1j * theta * np.array([1, -1, -1, 1])))


def _paulis() -> tuple:
    return (np.eye(2, dtype=np.complex128),
            np.array([[0, 1], [1, 0]], dtype=np.complex128),
            np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            np.diag([1, -1]).astype(np.complex128))


def get_depolarizing_kraus(p: float) -> list[np.ndarray]:
    """Single-qubit Kraus operators for (1-p)rho + p I/2."""
    p = _probability(p, "p")
    identity, x, y, z = _paulis()
    return [np.sqrt(1 - 3 * p / 4) * identity,
            np.sqrt(p) / 2 * x, np.sqrt(p) / 2 * y, np.sqrt(p) / 2 * z]


def get_two_qubit_depolarizing_kraus(p: float) -> list[np.ndarray]:
    """Sixteen Pauli Kraus operators for (1-p)rho + p I/4."""
    p = _probability(p, "p")
    paulis = _paulis()
    result = []
    for a, left in enumerate(paulis):
        for b, right in enumerate(paulis):
            coefficient = np.sqrt(1 - 15 * p / 16) if a == b == 0 else np.sqrt(p) / 4
            result.append(coefficient * np.kron(left, right))
    return result


def embed_operator_dd(matrix: np.ndarray, qubits: Sequence[int], n_qubits: int,
                      unique_table: UniqueTable) -> DDEdge:
    """Embed a local operator into arbitrary physical positions using DDs.

    qubits[0] is the local least significant qubit, as in Qiskit qargs.
    Convert only the local matrix to a DD, then recursively insert identity
    levels and route its leaf coefficients to the physical variable order.
    A memoized identity spine avoids visiting every full-system matrix entry.
    """
    n = _integer(n_qubits, "n_qubits", 1)
    targets = tuple(_integer(q, "target qubit") for q in qubits)
    if not targets or len(set(targets)) != len(targets) or any(q >= n for q in targets):
        raise ValueError("targets must be nonempty, distinct and inside the system")
    data = np.asarray(matrix, dtype=np.complex128)
    if data.shape != (1 << len(targets), 1 << len(targets)):
        raise ValueError("local matrix shape does not match target count")
    local = matrix_to_dd(data, unique_table)
    position = {q: bit for bit, q in enumerate(targets)}
    zero = DDEdge(0j, TERMINAL)

    def coefficient(row: int, col: int) -> DDEdge:
        edge = local
        weight = complex(edge.weight)
        for bit in range(len(targets) - 1, -1, -1):
            if weight == 0:
                return zero
            index = 2 * ((row >> bit) & 1) + ((col >> bit) & 1)
            edge = edge.node.edges[index]
            weight *= edge.weight
        return DDEdge(weight, TERMINAL)

    @lru_cache(maxsize=None)
    def build(var: int, row: int, col: int) -> DDEdge:
        if var < 0:
            return coefficient(row, col)
        if var not in position:
            diagonal = build(var - 1, row, col)
            children = (diagonal, zero, zero, diagonal)
        else:
            bit = position[var]
            children = tuple(build(var - 1, row | (r << bit), col | (c << bit))
                             for r, c in ((0, 0), (0, 1), (1, 0), (1, 1)))
        return unique_table.lookup_or_create(var, children)

    return build(n - 1, 0, 0)


def get_zz_crosstalk_dd(theta: float, qubits: Sequence[int], n_qubits: int,
                       unique_table: UniqueTable) -> DDEdge:
    if len(qubits) != 2:
        raise ValueError("ZZ requires exactly two qubits")
    return embed_operator_dd(get_zz_crosstalk_unitary(theta), qubits, n_qubits, unique_table)


def get_depolarizing_kraus_dd(p: float, qubits: Sequence[int], n_qubits: int,
                             unique_table: UniqueTable) -> list[DDEdge]:
    if len(qubits) == 1:
        matrices = get_depolarizing_kraus(p)
    elif len(qubits) == 2:
        matrices = get_two_qubit_depolarizing_kraus(p)
    else:
        raise ValueError("depolarizing noise requires one or two target qubits")
    return [embed_operator_dd(matrix, qubits, n_qubits, unique_table) for matrix in matrices]


def zero_density_dd(n_qubits: int, unique_table: UniqueTable) -> DDEdge:
    """Construct the all-zero basis density in O(n) nodes, without dense arrays."""
    n = _integer(n_qubits, "n_qubits", 1)
    root = DDEdge(1 + 0j, TERMINAL)
    zero = DDEdge(0j, TERMINAL)
    for var in range(n):
        root = unique_table.lookup_or_create(var, (root, zero, zero, zero))
    return root


def dd_trace(edge: DDEdge) -> complex:
    @lru_cache(maxsize=None)
    def trace_node(node) -> complex:
        if node is TERMINAL:
            return 1 + 0j
        return sum(child.weight * trace_node(child.node)
                   for child in (node.edges[0], node.edges[3]) if child.weight != 0)
    return complex(edge.weight * trace_node(edge.node))


def reachable_node_count(edge: DDEdge) -> int:
    visited = set()
    stack = [edge]
    while stack:
        current = stack.pop()
        if current.weight == 0 or current.node is TERMINAL or current.node in visited:
            continue
        visited.add(current.node)
        stack.extend(current.node.edges)
    return len(visited)


class CrosstalkDDSimulator:
    """Run a mapped unitary circuit with per-instruction hardware noise.

    Barriers are no-ops. Measurements, resets, delays, classical conditions,
    unbound parameters and gates wider than two qubits are rejected explicitly.
    Decompose larger gates and map/transpile before using this interface.
    Each run starts from |0><0| with fresh tables; no hidden state continuation.
    Global phase cancels in density evolution. Approximation may perturb trace,
    Hermiticity and positivity; diagnostics report trace without renormalizing.
    """

    def __init__(self, circuit: QuantumCircuit, topology: HardwareTopology,
                 epsilon: float = 1e-4, *, enforce_coupling: bool = True):
        if not isinstance(circuit, QuantumCircuit) or not isinstance(topology, HardwareTopology):
            raise TypeError("expected QuantumCircuit and HardwareTopology")
        if circuit.num_qubits != topology.n_qubits:
            raise ValueError("circuit width must equal hardware width, including idle spectators")
        epsilon = float(epsilon)
        if not np.isfinite(epsilon) or not 0 < epsilon < 1:
            raise ValueError("epsilon must be finite and in (0, 1)")
        self.circuit = circuit.copy()
        self.topology = topology
        self.epsilon = epsilon
        self.enforce_coupling = bool(enforce_coupling)
        self.n_qubits = topology.n_qubits
        self._instructions = self._compile()
        self._reset()

    def _compile(self) -> list:
        if self.circuit.num_parameters:
            raise ValueError("bind all circuit parameters before simulation")
        instructions = []
        for index, item in enumerate(self.circuit.data):
            operation = item.operation
            name = operation.name
            if item.clbits or getattr(operation, "condition", None) is not None:
                raise ValueError(f"instruction {index}: classical operations are unsupported")
            if name == "barrier":
                continue
            if name in {"measure", "reset", "delay", "initialize", "if_else", "while_loop", "for_loop", "switch_case"}:
                raise ValueError(f"instruction {index}: unsupported operation {name}")
            targets = tuple(self.circuit.find_bit(q).index for q in item.qubits)
            if len(targets) not in (1, 2):
                raise ValueError(f"instruction {index}: decompose {name} into one/two-qubit gates")
            if (len(targets) == 2 and self.enforce_coupling
                    and targets[1] not in self.topology.coupling_map[targets[0]]):
                raise ValueError(f"instruction {index}: gate {name}{targets} is not hardware-adjacent")
            try:
                matrix = np.asarray(Operator(operation).data, dtype=np.complex128)
            except (TypeError, ValueError, qiskit.exceptions.QiskitError) as error:
                raise ValueError(f"instruction {index}: {name} has no supported unitary matrix") from error
            if not np.all(np.isfinite(matrix)) or not np.allclose(
                    matrix.conj().T @ matrix, np.eye(matrix.shape[0]), atol=1e-12, rtol=1e-12):
                raise ValueError(f"instruction {index}: {name} is not unitary")
            pairs = tuple(sorted({tuple(sorted((q, neighbor))) for q in targets
                                  for neighbor in self.topology.neighbors(q)}))
            instructions.append((index, name, targets, matrix, pairs))
        return instructions

    def _reset(self) -> None:
        self.unique_table = UniqueTable(self.epsilon)
        self.compute_table = ComputeTable(self.epsilon)
        self.rho_edge = zero_density_dd(self.n_qubits, self.unique_table)
        self.history = []
        self._operator_cache = {}
        self._noise_cache = {}
        self._zz_cache = {}

    def run(self) -> DDEdge:
        self._reset()
        for index, name, targets, matrix, pairs in self._instructions:
            gate_key = (targets, matrix.shape, matrix.tobytes())
            if gate_key not in self._operator_cache:
                self._operator_cache[gate_key] = embed_operator_dd(
                    matrix, targets, self.n_qubits, self.unique_table)
            self.rho_edge = apply_unitary(self.rho_edge, self._operator_cache[gate_key],
                                          self.n_qubits, self.unique_table, self.compute_table)
            p = self.topology.p_depol if len(targets) == 1 else self.topology.p_cx_depol
            if p > 0:
                noise_key = (targets, p)
                if noise_key not in self._noise_cache:
                    self._noise_cache[noise_key] = get_depolarizing_kraus_dd(
                        p, targets, self.n_qubits, self.unique_table)
                self.rho_edge = apply_kraus_channel(
                    self.rho_edge, self._noise_cache[noise_key], self.n_qubits,
                    self.unique_table, self.compute_table)
            injected = []
            if self.topology.zz_theta != 0:
                for pair in pairs:
                    if pair not in self._zz_cache:
                        self._zz_cache[pair] = get_zz_crosstalk_dd(
                            self.topology.zz_theta, pair, self.n_qubits, self.unique_table)
                    self.rho_edge = apply_unitary(
                        self.rho_edge, self._zz_cache[pair], self.n_qubits,
                        self.unique_table, self.compute_table)
                    injected.append(list(pair))
            trace = dd_trace(self.rho_edge)
            self.history.append({"instruction": index, "gate": name, "qubits": list(targets),
                                 "depolarizing_strength": p,
                                 "kraus_count": (4 ** len(targets)) if p > 0 else 0,
                                 "zz_pairs": injected, "trace_real": trace.real,
                                 "trace_imag": trace.imag,
                                 "reachable_nodes": reachable_node_count(self.rho_edge),
                                 "retained_nodes": self.unique_table.get_active_node_count()})
        return self.rho_edge

    def save_history(self, path: str | os.PathLike) -> None:
        path = os.path.abspath(os.fspath(path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as stream:
            json.dump({"qiskit_version": qiskit.__version__, "epsilon": self.epsilon,
                       "hardware": self.topology.to_dict(), "instructions": self.history},
                      stream, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.ry(0.37, 2)
    simulator = CrosstalkDDSimulator(
        circuit, HardwareTopology.linear(3, p_depol=0.003, p_cx_depol=0.02, zz_theta=0.04),
        epsilon=1e-12)
    result = simulator.run()
    trace = dd_trace(result)
    if abs(trace - 1) > 1e-9:
        raise AssertionError(f"trace preservation failed: {trace}")
    print(json.dumps({"qiskit_version": qiskit.__version__, "trace_real": trace.real,
                      "trace_imag": trace.imag, "history": simulator.history}, indent=2))
