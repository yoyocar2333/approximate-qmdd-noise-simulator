"""Correctness tests for benchmark references, metrics and resource handling."""

import json
import os
import tempfile
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Operator, state_fidelity

from crosstalk_model import HardwareTopology
from run_benchmarks import (
    EXACT_EPSILON, add_compression_metrics, comparison_table,
    dense_depolarize, dense_unitary, evaluate_fidelity, generate_circuit,
    parse_arguments, route_circuit, run_isolated, save_json, simulate_dense,
)
from test_crosstalk_model import dense_reference


class BenchmarkTests(unittest.TestCase):
    def test_ghz_and_full_qft(self):
        for n in (2, 3, 4):
            ghz = generate_circuit("ghz", n)
            state = Operator(ghz).data[:, 0]
            expected = np.zeros(1 << n, dtype=complex)
            expected[0] = expected[-1] = 1 / np.sqrt(2)
            np.testing.assert_allclose(state, expected, atol=1e-14)
            qft = generate_circuit("qft", n)
            preparation = QuantumCircuit(n)
            preparation.h(0)
            preparation.t(0)
            preparation.h(0)
            indices = np.arange(1 << n)
            fourier = np.exp(2j * np.pi * indices[:, None] * indices[None, :] / (1 << n)) / np.sqrt(1 << n)
            np.testing.assert_allclose(Operator(qft).data, fourier @ Operator(preparation).data, atol=1e-13)

    def test_reproducibility_and_routing(self):
        first = generate_circuit("random_clifford_t", 4, 123, 3)
        self.assertEqual(first, generate_circuit("random_clifford_t", 4, 123, 3))
        self.assertNotEqual(first, generate_circuit("random_clifford_t", 4, 124, 3))
        self.assertIn("t", first.count_ops())
        hardware = HardwareTopology.linear(4)
        routed = route_circuit(first, hardware, 123)
        for item in routed.data:
            if len(item.qubits) == 2:
                a, b = [routed.find_bit(q).index for q in item.qubits]
                self.assertIn(b, hardware.neighbors(a))

    def test_dense_contractions_against_full_operators(self):
        rng = np.random.default_rng(44)
        raw = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
        rho = raw @ raw.conj().T
        rho /= np.trace(rho)
        for targets in ((0,), (2,), (0, 2), (2, 0), (1, 2)):
            d = 1 << len(targets)
            raw_gate = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
            gate, _ = np.linalg.qr(raw_gate)
            circuit = QuantumCircuit(3)
            circuit.unitary(gate, list(targets))
            full = Operator(circuit).data
            np.testing.assert_allclose(dense_unitary(rho, gate, targets, 3), full @ rho @ full.conj().T, atol=1e-13)
            output = dense_depolarize(rho, 0.3, targets, 3)
            self.assertLess(abs(np.trace(output) - 1), 1e-13)
            self.assertGreaterEqual(np.linalg.eigvalsh(output).min(), -1e-13)

    def test_dense_noise_against_independent_reference(self):
        circuit = QuantumCircuit(3)
        circuit.h(2)
        circuit.cx(2, 0)
        circuit.ry(0.31, 1)
        circuit.cp(0.27, 1, 0)
        topology = HardwareTopology.ring(3, p_depol=0.041, p_cx_depol=0.077, zz_theta=0.091)
        np.testing.assert_allclose(simulate_dense(circuit, topology), dense_reference(circuit, topology), atol=1e-13)

    def test_fidelity_against_qiskit(self):
        rng = np.random.default_rng(37)
        states = []
        for index in range(2):
            raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
            rho = raw @ raw.conj().T
            states.append(rho / np.trace(rho))
        result = evaluate_fidelity(*states)
        self.assertTrue(result["is_physical"])
        self.assertAlmostEqual(result["fidelity"], state_fidelity(DensityMatrix(states[0]), DensityMatrix(states[1])), places=12)
        self.assertAlmostEqual(evaluate_fidelity(np.diag([1, 0]), np.diag([0, 1]))["fidelity"], 0)
        self.assertAlmostEqual(evaluate_fidelity(np.eye(4) / 4, np.eye(4) / 4)["fidelity"], 1)

    def test_nonphysical_states_are_not_silent_fidelity_successes(self):
        for bad in (np.diag([1.1, -0.1]), np.eye(2), np.array([[1, 0.2j], [0, 0]])):
            metrics = evaluate_fidelity(np.eye(2) / 2, bad)
            self.assertFalse(metrics["is_physical"])
            self.assertIsNone(metrics["fidelity"])
            self.assertIsNone(metrics["infidelity"])
            self.assertGreater(metrics["projection_distance_fro"], 0)
            self.assertTrue(0 <= metrics["projected_fidelity"] <= 1)

    def test_json_table_and_compression(self):
        exact = {"case_id": "ghz3", "family": "ghz", "n_qubits": 3, "seed": 2026, "repeat": 1,
                 "method": "exact_dd", "epsilon": EXACT_EPSILON, "status": "ok",
                 "final_state_nodes": 20, "peak_active_nodes": 100, "simulation_seconds": 0.5}
        approximate = {**exact, "method": "approx_dd", "epsilon": 1e-4,
                       "final_state_nodes": 5, "peak_active_nodes": 20}
        rows = [exact, approximate]
        add_compression_metrics(rows)
        self.assertEqual(approximate["final_state_node_reduction"], 0.75)
        self.assertEqual(approximate["final_state_compression_factor"], 4)
        self.assertIn("| Circuit", comparison_table(rows))
        self.assertIn("\\begin{tabular}", comparison_table(rows, True))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "results.json")
            save_json(path, {"runs": rows})
            with open(path, encoding="utf-8") as stream:
                self.assertEqual(json.load(stream)["runs"], rows)

    def test_isolated_run_and_budget_reporting(self):
        config = parse_arguments(["--max-qubits", "3", "--families", "ghz", "--timeout-seconds", "15"])
        circuit = generate_circuit("ghz", 3)
        topology = HardwareTopology.linear(3)
        result = run_isolated(circuit, topology, "exact_dd", EXACT_EPSILON, config)
        self.assertEqual(result["status"], "ok", result)
        self.assertGreaterEqual(result["peak_active_nodes"], result["final_state_nodes"])
        np.testing.assert_allclose(result["matrix"], simulate_dense(circuit, topology), atol=1e-10)
        limited = run_isolated(circuit, topology, "exact_dd", EXACT_EPSILON, {**config, "max_dd_nodes": 1})
        self.assertEqual(limited["status"], "resource_limit")
        self.assertNotIn("matrix", limited)
        timeout = run_isolated(circuit, topology, "dense", None, {**config, "timeout_seconds": 0.001})
        self.assertEqual(timeout["status"], "timeout")


if __name__ == "__main__":
    unittest.main(verbosity=2)
