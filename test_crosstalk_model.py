"""Independent dense-reference and structural tests for phase three.

Run: python -m unittest -v test_crosstalk_model
Dense system matrices occur only in this test module.
"""

import itertools
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Operator, Statevector
from qiskit.transpiler import CouplingMap

import dd_core
import dd_ops
from dd_core import UniqueTable, dd_to_matrix
from crosstalk_model import (
    CrosstalkDDSimulator, HardwareTopology, dd_trace, embed_operator_dd,
    get_depolarizing_kraus, get_depolarizing_kraus_dd,
    get_two_qubit_depolarizing_kraus, get_zz_crosstalk_dd,
    get_zz_crosstalk_unitary, reachable_node_count, zero_density_dd,
)


def dense_depolarize(rho: np.ndarray, targets: tuple, p: float) -> np.ndarray:
    """Partial-trace replacement formula, independent of Pauli Kraus construction."""
    mask = sum(1 << q for q in targets)
    mixed = np.zeros_like(rho)
    dimension = 1 << len(targets)
    for row in range(rho.shape[0]):
        for col in range(rho.shape[1]):
            if (row & mask) != (col & mask):
                continue
            subtotal = 0j
            for local in range(dimension):
                physical = sum(((local >> bit) & 1) << q for bit, q in enumerate(targets))
                subtotal += rho[(row & ~mask) | physical, (col & ~mask) | physical]
            mixed[row, col] = subtotal / dimension
    return (1 - p) * rho + p * mixed


def dense_reference(circuit: QuantumCircuit, topology: HardwareTopology) -> np.ndarray:
    n = circuit.num_qubits
    rho = np.zeros((1 << n, 1 << n), dtype=np.complex128)
    rho[0, 0] = 1
    for item in circuit.data:
        if item.operation.name == "barrier":
            continue
        targets = tuple(circuit.find_bit(q).index for q in item.qubits)
        unitary_circuit = QuantumCircuit(n)
        unitary_circuit.append(item.operation, list(targets))
        unitary = Operator(unitary_circuit).data
        rho = unitary @ rho @ unitary.conj().T
        p = topology.p_depol if len(targets) == 1 else topology.p_cx_depol
        rho = dense_depolarize(rho, targets, p)
        active = set(targets)
        for a, b in topology.edges:
            if a not in active and b not in active:
                continue
            signs = np.array([1 if ((basis >> a) & 1) == ((basis >> b) & 1) else -1
                              for basis in range(1 << n)])
            phase = np.exp(-1j * topology.zz_theta * signs)
            rho = phase[:, None] * rho * phase.conj()[None, :]
    return rho


class CrosstalkTests(unittest.TestCase):
    def test_topologies_and_json(self):
        linear = HardwareTopology.linear(4, p_depol=0.02, p_cx_depol=0.04, zz_theta=-0.1)
        self.assertEqual(linear.neighbors(1), (0, 2))
        self.assertEqual(HardwareTopology.ring(4).neighbors(0), (1, 3))
        self.assertEqual(HardwareTopology(3, {0: [1]}).neighbors(1), (0,))
        self.assertEqual(HardwareTopology(3, {0: [1]}).neighbors(2), ())
        heavy = HardwareTopology.heavy_hex(3)
        self.assertEqual(heavy.n_qubits, 19)
        self.assertLessEqual(max(map(len, heavy.coupling_map.values())), 3)
        coupling = CouplingMap([[0, 1], [1, 2]])
        self.assertEqual(HardwareTopology.from_coupling_map(coupling).neighbors(1), (0, 2))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "hardware.json")
            linear.to_json(path)
            loaded = HardwareTopology.from_json(path)
            self.assertEqual(linear.to_dict(), loaded.to_dict())

    def test_invalid_topology_and_parameters(self):
        for pairs in ([(0, 0)], [(0, 3)], [(0, -1)]):
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                HardwareTopology(3, pairs)
        for value in (-0.01, 1.1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                get_depolarizing_kraus(value)
        with self.assertRaises(ValueError):
            HardwareTopology.heavy_hex(2)
        with self.assertRaises(ValueError):
            get_zz_crosstalk_unitary(float("nan"))
        with self.assertRaises(ValueError):
            embed_operator_dd(np.eye(4), [0, 0], 3, UniqueTable())

    def test_kraus_completeness_and_channel_convention(self):
        rng = np.random.default_rng(71)
        for dimension, factory in ((2, get_depolarizing_kraus),
                                   (4, get_two_qubit_depolarizing_kraus)):
            for p in (0, 0.13, 1):
                with self.subTest(dimension=dimension, p=p):
                    operators = factory(p)
                    self.assertEqual(len(operators), dimension * dimension)
                    total = sum((e.conj().T @ e for e in operators),
                                np.zeros((dimension, dimension), dtype=np.complex128))
                    np.testing.assert_allclose(total, np.eye(dimension), atol=1e-14)
                    raw = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
                    rho = raw @ raw.conj().T
                    rho /= np.trace(rho)
                    evolved = sum((e @ rho @ e.conj().T for e in operators), np.zeros_like(rho))
                    np.testing.assert_allclose(evolved, (1 - p) * rho + p * np.eye(dimension) / dimension,
                                               atol=1e-14)

    def test_zz_angle_and_noncontiguous_embedding(self):
        theta = 0.173
        local = get_zz_crosstalk_unitary(theta)
        np.testing.assert_allclose(local.conj().T @ local, np.eye(4), atol=1e-14)
        circuit = QuantumCircuit(3)
        circuit.rzz(2 * theta, 2, 0)
        edge = get_zz_crosstalk_dd(theta, [2, 0], 3, UniqueTable(1e-12))
        np.testing.assert_allclose(dd_to_matrix(edge, 3), Operator(circuit).data, atol=1e-11)

    def test_embedding_all_target_orders(self):
        rng = np.random.default_rng(123)
        for width in (1, 2):
            raw = rng.normal(size=(1 << width, 1 << width)) + 1j * rng.normal(size=(1 << width, 1 << width))
            unitary, _ = np.linalg.qr(raw)
            for targets in itertools.permutations(range(3), width):
                with self.subTest(targets=targets):
                    circuit = QuantumCircuit(3)
                    circuit.unitary(unitary, list(targets))
                    edge = embed_operator_dd(unitary, targets, 3, UniqueTable(1e-12))
                    np.testing.assert_allclose(dd_to_matrix(edge, 3), Operator(circuit).data, atol=1e-11)
        for targets in ((0, 2), (2, 0)):
            circuit = QuantumCircuit(3)
            circuit.cx(*targets)
            edge = embed_operator_dd(Operator(CXGate()).data, targets, 3, UniqueTable(1e-12))
            np.testing.assert_allclose(dd_to_matrix(edge, 3), Operator(circuit).data, atol=1e-12)

    def test_noise_free_qiskit_statevector(self):
        a, b = QuantumRegister(1, "a"), QuantumRegister(2, "b")
        circuit = QuantumCircuit(a, b)
        circuit.global_phase = 0.41
        circuit.h(b[1])
        circuit.s(a[0])
        circuit.cx(b[1], a[0])
        circuit.ry(0.53, b[0])
        circuit.cx(a[0], b[0])
        topology = HardwareTopology.ring(3, p_depol=0, p_cx_depol=0, zz_theta=0)
        state = Statevector.from_instruction(circuit).data
        output = CrosstalkDDSimulator(circuit, topology, 1e-12).run()
        np.testing.assert_allclose(dd_to_matrix(output, 3), np.outer(state, state.conj()), atol=1e-10)

    def test_noisy_reference_trace_hermiticity_positivity(self):
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.ry(0.41, 2)
        circuit.cx(2, 0)
        circuit.s(1)
        circuit.rx(-0.29, 1)
        circuit.cx(0, 1)
        topology = HardwareTopology.ring(3, p_depol=0.037, p_cx_depol=0.081, zz_theta=0.067)
        output = CrosstalkDDSimulator(circuit, topology, 1e-12).run()
        actual, expected = dd_to_matrix(output, 3), dense_reference(circuit, topology)
        error = float(np.max(np.abs(actual - expected)))
        self.assertLess(error, 1e-9)
        self.assertLess(abs(dd_trace(output) - 1), 1e-9)
        np.testing.assert_allclose(actual, actual.conj().T, atol=1e-9)
        self.assertGreaterEqual(float(np.linalg.eigvalsh((actual + actual.conj().T) / 2).min()), -1e-9)
        print(json.dumps({"dense_reference_max_error": error, "qiskit_version": qiskit.__version__}))

    def test_default_epsilon_against_reference(self):
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.ry(0.37, 2)
        topology = HardwareTopology.linear(3, p_depol=0.003, p_cx_depol=0.02, zz_theta=0.04)
        actual = dd_to_matrix(CrosstalkDDSimulator(circuit, topology).run(), 3)
        error = float(np.max(np.abs(actual - dense_reference(circuit, topology))))
        self.assertLess(error, 5e-3)
        print(json.dumps({"epsilon": 1e-4, "demo_max_error": error}))

    def test_idle_spectator_and_pair_deduplication(self):
        circuit = QuantumCircuit(3)
        circuit.h(2)
        circuit.x(0)
        circuit.x(0)
        topology = HardwareTopology(3, [(0, 2)], p_depol=0, p_cx_depol=0, zz_theta=0.16)
        simulator = CrosstalkDDSimulator(circuit, topology, 1e-12)
        output = dd_to_matrix(simulator.run(), 3)
        self.assertEqual(simulator.history[-1]["qubits"], [0])
        self.assertEqual(simulator.history[-1]["zz_pairs"], [[0, 2]])
        np.testing.assert_allclose(output, dense_reference(circuit, topology), atol=1e-10)
        quiet = HardwareTopology(3, [(0, 2)], p_depol=0, p_cx_depol=0, zz_theta=0)
        baseline = dense_reference(circuit, quiet)
        spectator = np.array([[sum(output[(r << 2) | k, (c << 2) | k] for k in range(4))
                               for c in range(2)] for r in range(2)])
        baseline_spectator = np.array([[sum(baseline[(r << 2) | k, (c << 2) | k] for k in range(4))
                                        for c in range(2)] for r in range(2)])
        self.assertGreater(float(np.max(np.abs(spectator - baseline_spectator))), 0.05)
        pair_circuit = QuantumCircuit(3)
        pair_circuit.cx(0, 1)
        model = CrosstalkDDSimulator(pair_circuit, HardwareTopology.ring(3), 1e-12)
        model.run()
        self.assertEqual(model.history[0]["zz_pairs"], [[0, 1], [0, 2], [1, 2]])
        self.assertEqual(model.history[0]["kraus_count"], 16)

    def test_full_depolarization_on_gate_subsystem(self):
        for width in (1, 2):
            circuit = QuantumCircuit(3)
            if width == 1:
                circuit.x(2)
            else:
                circuit.cx(2, 0)
            topology = HardwareTopology.ring(3, p_depol=1, p_cx_depol=1, zz_theta=0)
            actual = dd_to_matrix(CrosstalkDDSimulator(circuit, topology, 1e-12).run(), 3)
            np.testing.assert_allclose(actual, dense_reference(circuit, topology), atol=1e-12)
        edges = get_depolarizing_kraus_dd(0, [0], 3, UniqueTable(1e-12))
        self.assertEqual(len(edges), 4)
        self.assertEqual(sum(edge.weight != 0 for edge in edges), 1)

    def test_repeat_run_barrier_and_json_history(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.barrier()
        circuit.cx(0, 1)
        simulator = CrosstalkDDSimulator(circuit, HardwareTopology.linear(2), 1e-12)
        first = dd_to_matrix(simulator.run(), 2)
        second = dd_to_matrix(simulator.run(), 2)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(simulator.history), 2)
        self.assertEqual([item["instruction"] for item in simulator.history], [0, 2])
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "history.json")
            simulator.save_history(path)
            with open(path, encoding="utf-8") as stream:
                data = json.load(stream)
            self.assertEqual(data["instructions"], simulator.history)

    def test_reject_unsupported_instructions_and_unmapped_gates(self):
        for name in ("measure", "reset", "delay", "parameter", "ccx", "condition"):
            circuit = QuantumCircuit(3, 1)
            if name == "measure":
                circuit.measure(0, 0)
            elif name == "reset":
                circuit.reset(0)
            elif name == "delay":
                circuit.delay(10, 0)
            elif name == "parameter":
                circuit.rx(Parameter("theta"), 0)
            elif name == "ccx":
                circuit.ccx(0, 1, 2)
            else:
                with circuit.if_test((circuit.clbits[0], 1)):
                    circuit.x(0)
            with self.subTest(name=name), self.assertRaises(ValueError):
                CrosstalkDDSimulator(circuit, HardwareTopology.linear(3))
        circuit = QuantumCircuit(3)
        circuit.cx(0, 2)
        with self.assertRaises(ValueError):
            CrosstalkDDSimulator(circuit, HardwareTopology.linear(3))
        relaxed = CrosstalkDDSimulator(circuit, HardwareTopology.linear(3), enforce_coupling=False)
        self.assertIsNotNone(relaxed.run())
        with self.assertRaises(ValueError):
            CrosstalkDDSimulator(QuantumCircuit(2), HardwareTopology.linear(3))

    def test_simulation_does_not_expand_density(self):
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        with patch.object(dd_core, "dd_to_matrix", side_effect=AssertionError("dense conversion")), \
             patch.object(dd_ops, "dd_to_matrix", side_effect=AssertionError("dense conversion")):
            output = CrosstalkDDSimulator(circuit, HardwareTopology.linear(3), 1e-12).run()
            self.assertLess(abs(dd_trace(output) - 1), 1e-9)

    def test_heavy_hex_idle_spectators_without_dense_allocations(self):
        topology = HardwareTopology.heavy_hex(3, p_depol=0, p_cx_depol=0, zz_theta=0.03)
        table = UniqueTable(1e-12)
        initial = zero_density_dd(topology.n_qubits, table)
        self.assertEqual(reachable_node_count(initial), 19)
        circuit = QuantumCircuit(topology.n_qubits)
        circuit.h(0)
        simulator = CrosstalkDDSimulator(circuit, topology, 1e-12)
        output = simulator.run()
        self.assertLess(abs(dd_trace(output) - 1), 1e-9)
        self.assertEqual(simulator.history[0]["zz_pairs"], [[0, neighbor] for neighbor in topology.neighbors(0)])
        self.assertLess(reachable_node_count(output), 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
