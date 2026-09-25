"""Reproducible dense/DD benchmarks with checkpointed JSON and IEEE-size figures.

Run python run_benchmarks.py --help for all controls. Default n=3..8.
Exact DD means epsilon=1e-15 in the existing floating-point engine.
Raw fidelity is reported only for physical density matrices. A separately
labelled PSD-projected fidelity is a diagnostic, never a simulation correction.
"""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter
import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap
from threadpoolctl import threadpool_limits

import crosstalk_model as crosstalk
import dd_core
import dd_ops
from crosstalk_model import HardwareTopology, CrosstalkDDSimulator, reachable_node_count
from dd_core import dd_to_matrix


FAMILIES = ("ghz", "qft", "random_clifford_t")
FAMILY_LABELS = {"ghz": "GHZ", "qft": "QFT", "random_clifford_t": "Clifford+T"}
EXACT_EPSILON = 1e-15


def generate_circuit(family: str, n_qubits: int, seed: int = 2026,
                     random_depth: int = 3) -> QuantumCircuit:
    """Generate deterministic benchmark circuits before physical routing."""
    if n_qubits < 1 or random_depth < 1:
        raise ValueError("n_qubits and random_depth must be positive")
    circuit = QuantumCircuit(n_qubits, name=f"{family}_{n_qubits}_{seed}")
    if family == "ghz":
        circuit.h(0)
        for qubit in range(1, n_qubits):
            circuit.cx(qubit - 1, qubit)
    elif family == "qft":
        # A documented H-T-H input preparation avoids QFT on the trivial zero input.
        circuit.h(0)
        circuit.t(0)
        circuit.h(0)
        for target in reversed(range(n_qubits)):
            circuit.h(target)
            for control in reversed(range(target)):
                circuit.cp(np.pi / (2 ** (target - control)), control, target)
        for left in range(n_qubits // 2):
            circuit.swap(left, n_qubits - left - 1)
    elif family == "random_clifford_t":
        rng = np.random.default_rng(np.random.SeedSequence([seed, n_qubits]))
        circuit.h(0)
        choices = ("h", "s", "sdg", "x", "y", "z", "t", "tdg")
        for layer in range(random_depth):
            for qubit in range(n_qubits):
                getattr(circuit, choices[int(rng.integers(len(choices)))])(qubit)
            circuit.t(int(rng.integers(n_qubits)))
            order = rng.permutation(n_qubits).tolist()
            for offset in range(0, n_qubits - 1, 2):
                circuit.cx(order[offset], order[offset + 1])
    else:
        raise ValueError(f"unknown circuit family: {family}")
    return circuit


def make_topology(n_qubits: int, config: dict) -> HardwareTopology:
    parameters = {name: config[name] for name in ("p_depol", "p_cx_depol", "zz_theta")}
    if config["topology"] == "linear":
        return HardwareTopology.linear(n_qubits, **parameters)
    if config["topology"] == "ring":
        return HardwareTopology.ring(n_qubits, **parameters)
    return HardwareTopology(n_qubits, [(a, b) for a in range(n_qubits)
                                      for b in range(a + 1, n_qubits)], **parameters)


def route_circuit(circuit: QuantumCircuit, topology: HardwareTopology,
                  seed: int) -> QuantumCircuit:
    coupling = CouplingMap()
    for qubit in range(topology.n_qubits):
        coupling.add_physical_qubit(qubit)
    for a, b in topology.edges:
        coupling.add_edge(a, b)
        coupling.add_edge(b, a)
    return transpile(circuit, coupling_map=coupling,
                     basis_gates=["h", "s", "sdg", "t", "tdg", "x", "y", "z",
                                  "rx", "ry", "rz", "cx", "cp", "swap", "id"],
                     initial_layout=list(range(topology.n_qubits)),
                     optimization_level=0, seed_transpiler=seed)


def _subsystem_view(rho: np.ndarray, targets: tuple, n_qubits: int) -> tuple:
    # Local matrix order is targets[k-1] through targets[0], following Qiskit.
    active = [n_qubits - 1 - q for q in reversed(targets)]
    other = [axis for axis in range(n_qubits) if axis not in active]
    axes = active + other + [axis + n_qubits for axis in active + other]
    dim = 1 << len(targets)
    remainder = 1 << (n_qubits - len(targets))
    view = rho.reshape([2] * (2 * n_qubits)).transpose(axes).reshape(dim, remainder, dim, remainder)
    return view, np.argsort(axes)


def _restore_subsystem(view: np.ndarray, inverse_axes: np.ndarray,
                       n_qubits: int) -> np.ndarray:
    return view.reshape([2] * (2 * n_qubits)).transpose(inverse_axes).reshape(1 << n_qubits, 1 << n_qubits)


def dense_unitary(rho: np.ndarray, unitary: np.ndarray, targets: tuple,
                  n_qubits: int) -> np.ndarray:
    """Exact dense-array contraction without constructing a full gate matrix."""
    view, inverse = _subsystem_view(rho, targets, n_qubits)
    evolved = np.einsum("ai,irjs,bj->arbs", unitary, view, unitary.conj(), optimize=True)
    return _restore_subsystem(evolved, inverse, n_qubits)


def dense_depolarize(rho: np.ndarray, p: float, targets: tuple,
                     n_qubits: int) -> np.ndarray:
    """Independent partial-trace replacement, not a reuse of DD Kraus code."""
    if p == 0:
        return rho
    view, inverse = _subsystem_view(rho, targets, n_qubits)
    reduced = np.trace(view, axis1=0, axis2=2)
    dim = view.shape[0]
    replacement = np.einsum("ab,rs->arbs", np.eye(dim) / dim, reduced)
    return _restore_subsystem((1 - p) * view + p * replacement, inverse, n_qubits)


def simulate_dense(circuit: QuantumCircuit, topology: HardwareTopology) -> np.ndarray:
    n = circuit.num_qubits
    rho = np.zeros((1 << n, 1 << n), dtype=np.complex128)
    rho[0, 0] = 1
    phase_cache = {}
    indices = np.arange(1 << n)
    for item in circuit.data:
        if item.operation.name == "barrier":
            continue
        targets = tuple(circuit.find_bit(q).index for q in item.qubits)
        rho = dense_unitary(rho, Operator(item.operation).data, targets, n)
        p = topology.p_depol if len(targets) == 1 else topology.p_cx_depol
        rho = dense_depolarize(rho, p, targets, n)
        if topology.zz_theta != 0:
            for a, b in topology.edges:
                if a not in targets and b not in targets:
                    continue
                if (a, b) not in phase_cache:
                    sign = 1 - 2 * (((indices >> a) ^ (indices >> b)) & 1)
                    phase_cache[(a, b)] = np.exp(-1j * topology.zz_theta * sign)
                phase = phase_cache[(a, b)]
                rho = phase[:, None] * rho * phase.conj()[None, :]
    return rho


def _density_analysis(matrix: np.ndarray, tolerance: float) -> tuple:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.all(np.isfinite(matrix)):
        raise ValueError("density matrix must be finite and square")
    trace = np.trace(matrix)
    hermitian = (matrix + matrix.conj().T) / 2
    eigenvalues, vectors = np.linalg.eigh(hermitian)
    hermiticity_error = float(np.linalg.norm(matrix - matrix.conj().T, ord="fro"))
    trace_error = float(abs(trace - 1))
    minimum = float(eigenvalues.min())
    valid = trace_error <= tolerance and hermiticity_error <= tolerance and minimum >= -tolerance
    # Euclidean projection of eigenvalues onto the probability simplex.
    sorted_values = np.sort(eigenvalues)[::-1]
    thresholds = (np.cumsum(sorted_values) - 1) / np.arange(1, len(sorted_values) + 1)
    last = int(np.flatnonzero(sorted_values > thresholds)[-1])
    projected_values = np.maximum(eigenvalues - thresholds[last], 0)
    projected = (vectors * projected_values) @ vectors.conj().T
    diagnostics = {"is_physical": bool(valid), "trace_real": float(trace.real),
                   "trace_imag": float(trace.imag), "trace_error": trace_error,
                   "hermiticity_error_fro": hermiticity_error, "minimum_eigenvalue": minimum,
                   "negative_eigenvalue_mass": float(-np.minimum(eigenvalues, 0).sum()),
                   "projection_distance_fro": float(np.linalg.norm(projected - matrix, ord="fro"))}
    return diagnostics, eigenvalues, vectors, projected


def _fidelity_from_root(reference_root: np.ndarray, candidate: np.ndarray) -> float:
    middle = reference_root @ candidate @ reference_root
    eigenvalues = np.linalg.eigvalsh((middle + middle.conj().T) / 2)
    return float(np.square(np.sqrt(np.maximum(eigenvalues, 0)).sum()))


def evaluate_fidelity(rho_exact: np.ndarray, rho_dd: np.ndarray,
                      tolerance: float = 1e-8) -> dict:
    """Uhlmann squared fidelity, plus explicit raw-state validity diagnostics."""
    if rho_exact.shape != rho_dd.shape:
        raise ValueError("reference and candidate dimensions differ")
    reference, values, vectors, _ = _density_analysis(rho_exact, tolerance)
    if not reference["is_physical"]:
        raise ValueError("dense reference is not a physical density matrix")
    candidate, _, _, projected = _density_analysis(rho_dd, tolerance)
    root = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.conj().T
    raw = _fidelity_from_root(root, (rho_dd + rho_dd.conj().T) / 2) if candidate["is_physical"] else None
    projected_raw = _fidelity_from_root(root, projected)
    # Clip only the reported fidelity's floating-point excursions; preserve raw value.
    fidelity = float(np.clip(raw, 0, 1)) if raw is not None else None
    projected_fidelity = float(np.clip(projected_raw, 0, 1))
    return {**candidate, "fidelity_unclipped": raw, "fidelity": fidelity,
            "infidelity": 1 - fidelity if fidelity is not None else None,
            "projected_fidelity": projected_fidelity,
            "projected_infidelity": 1 - projected_fidelity,
            "max_element_error": float(np.max(np.abs(rho_exact - rho_dd))),
            "matrix_error_fro": float(np.linalg.norm(rho_exact - rho_dd, ord="fro"))}


def _worker(connection, circuit, topology, method, epsilon, limits):
    """Isolated worker: bounded tables prevent one case exhausting the batch."""
    simulator = None
    start = time.perf_counter()
    try:
        topology = HardwareTopology(*topology)
        with threadpool_limits(limits=limits["blas_threads"]):
            start = time.perf_counter()
            if method == "dense":
                matrix = simulate_dense(circuit, topology)
                elapsed = time.perf_counter() - start
                result = {"status": "ok", "simulation_seconds": elapsed,
                          "export_seconds": 0.0, "matrix": matrix,
                          "dense_matrix_bytes": int(matrix.nbytes)}
            else:
                class BoundedUniqueTable(dd_core.UniqueTable):
                    def lookup_or_create(self, var, edges):
                        if self.get_active_node_count() >= limits["max_dd_nodes"]:
                            raise MemoryError("configured DD node budget reached")
                        return super().lookup_or_create(var, edges)

                class BoundedComputeTable(dd_ops.ComputeTable):
                    def put(self, key, result):
                        if len(self._cache) >= limits["max_cache_entries"]:
                            raise MemoryError("configured compute-cache budget reached")
                        return super().put(key, result)

                crosstalk.UniqueTable = BoundedUniqueTable
                crosstalk.ComputeTable = BoundedComputeTable
                simulator = CrosstalkDDSimulator(circuit, topology, epsilon)
                edge = simulator.run()
                elapsed = time.perf_counter() - start
                export_start = time.perf_counter()
                matrix = dd_to_matrix(edge, circuit.num_qubits)
                export_seconds = time.perf_counter() - export_start
                retained = simulator.unique_table.get_active_node_count()
                final_nodes = reachable_node_count(edge)
                result = {"status": "ok", "simulation_seconds": elapsed,
                          "export_seconds": export_seconds, "matrix": matrix,
                          "peak_active_nodes": retained, "final_active_nodes": retained,
                          "final_state_nodes": final_nodes,
                          "peak_state_nodes_at_instruction_boundaries": max(
                              [circuit.num_qubits] + [h["reachable_nodes"] for h in simulator.history]),
                          "compute_cache_entries": len(simulator.compute_table._cache),
                          "instruction_history": simulator.history}
            connection.send(result)
    except Exception as error:
        connection.send({"status": "resource_limit" if isinstance(error, MemoryError) else "error",
                         "elapsed_before_failure_seconds": time.perf_counter() - start,
                         "error": f"{type(error).__name__}: {error}",
                         "traceback": traceback.format_exc(),
                         "retained_nodes_at_failure": simulator.unique_table.get_active_node_count()
                         if simulator is not None else None})
    finally:
        connection.close()


def run_isolated(circuit: QuantumCircuit, topology: HardwareTopology,
                  method: str, epsilon: float | None, config: dict) -> dict:
    context = mp.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    hardware = (topology.n_qubits, topology.edges, topology.p_depol,
                topology.p_cx_depol, topology.zz_theta)
    worker = context.Process(target=_worker, args=(send, circuit, hardware, method, epsilon, config))
    start = time.perf_counter()
    worker.start()
    send.close()
    try:
        if receive.poll(config["timeout_seconds"]):
            try:
                result = receive.recv()
            except EOFError:
                result = {"status": "worker_failed", "error": "worker exited without a result"}
        else:
            result = {"status": "timeout", "error": "per-run wall-clock limit reached"}
        worker.join(timeout=0.2)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=2)
        if worker.is_alive():
            worker.kill()
            worker.join()
        result["worker_wall_seconds"] = time.perf_counter() - start
        result["worker_exit_code"] = worker.exitcode
        return result
    finally:
        receive.close()
        if worker.is_alive():
            worker.terminate()
            worker.join()


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _method_label(row: dict) -> str:
    if row["method"] == "dense":
        return "Dense"
    if row["method"] == "exact_dd":
        return "Exact DD (1e-15)"
    return f"Approx ({row['epsilon']:.0e})"


def comparison_table(runs: list[dict], latex: bool = False) -> str:
    headers = ["Circuit", "n", "Seed", "Rep", "Method", "Status", "Time (s)",
               "Peak retained", "Final state", "Raw 1-F", "Projected 1-F"]
    rows = []
    for row in runs:
        fidelity = row.get("accuracy", {})
        rows.append([FAMILY_LABELS[row["family"]], str(row["n_qubits"]), str(row["seed"]),
                     str(row["repeat"]), _method_label(row), row["status"],
                     f"{row['simulation_seconds']:.6f}" if "simulation_seconds" in row else "--",
                     str(row.get("peak_active_nodes", "--")), str(row.get("final_state_nodes", "--")),
                     f"{fidelity['infidelity']:.3e}" if fidelity.get("infidelity") is not None else "invalid" if fidelity else "--",
                     f"{fidelity['projected_infidelity']:.3e}" if fidelity else "--"])
    if latex:
        def escape(text):
            return text.replace("_", "\\_").replace("%", "\\%")
        lines = ["\\begin{tabular}{lrrrllrrrrr}", "\\hline",
                 " & ".join(headers) + r" \\", "\\hline"]
        lines.extend(" & ".join(escape(cell) for cell in row) + r" \\" for row in rows)
        return "\n".join(lines + ["\\hline", "\\end{tabular}"])
    widths = [max(len(headers[i]), max((len(row[i]) for row in rows), default=0)) for i in range(len(headers))]
    def line(row):
        return "| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |"
    return "\n".join([line(headers), line(["-" * width for width in widths])] + [line(row) for row in rows])


def add_compression_metrics(runs: list[dict]) -> None:
    exact = {(r["case_id"], r["repeat"]): r for r in runs
             if r["method"] == "exact_dd" and r["status"] == "ok"}
    for row in runs:
        base = exact.get((row["case_id"], row["repeat"]))
        if row["method"] != "approx_dd" or row["status"] != "ok" or base is None:
            continue
        for name, field in (("final_state", "final_state_nodes"), ("peak_active", "peak_active_nodes")):
            denominator = base[field]
            row[name + "_node_reduction"] = 1 - row[field] / denominator if denominator else None
            row[name + "_compression_factor"] = denominator / row[field] if row[field] else None


def plot_results(payload: dict, output_dir: str) -> None:
    """IEEE two-column width, readable typography, 600 dpi PNG and vector PDF."""
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "axes.linewidth": 0.7, "lines.linewidth": 1.2,
                         "lines.markersize": 4, "pdf.fonttype": 42, "ps.fonttype": 42})
    config = payload["config"]
    families = config["families"]
    runs = payload["runs"]
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"]
    styles = ["--", "-.", ":", "--", "-."]
    epsilons = config["epsilons"]
    methods = [("exact_dd", EXACT_EPSILON, "Exact DD", "#222222", "-", "o")]
    for i, epsilon in enumerate(epsilons):
        methods.append(("approx_dd", epsilon, f"epsilon={epsilon:.0e}",
                        colors[i % len(colors)], styles[i % len(styles)], ("s", "^", "D", "v", "P")[i % 5]))
    figure, axes = plt.subplots(1, len(families), figsize=(7.16, 3.05), squeeze=False)
    for index, family in enumerate(families):
        axis = axes[0, index]
        for method, epsilon, label, color, linestyle, marker in methods:
            xs, ys = [], []
            for n in range(config["min_qubits"], config["max_qubits"] + 1):
                values = [r["peak_active_nodes"] for r in runs if r["family"] == family
                          and r["n_qubits"] == n and r["method"] == method
                          and r["epsilon"] == epsilon and r["status"] == "ok"]
                xs.append(n)
                ys.append(statistics.median(values) if values else np.nan)
            axis.plot(xs, ys, label=label, color=color, linestyle=linestyle, marker=marker)
        axis.set_title(FAMILY_LABELS[family])
        axis.set_xlabel("Qubits")
        axis.set_yscale("log")
        axis.set_xticks(range(config["min_qubits"], config["max_qubits"] + 1))
        axis.grid(True, alpha=0.2, linewidth=0.5)
        if index == 0:
            axis.set_ylabel("Peak retained DD nodes")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)
    figure.text(0.5, 0.015, "Medians over seeds/repeats; missing points = failed or limited runs.", ha="center", fontsize=7)
    figure.tight_layout(rect=(0, 0.05, 1, 0.86))
    for extension in ("png", "pdf"):
        figure.savefig(os.path.join(output_dir, "node_compression." + extension), dpi=600)
    plt.close(figure)

    figure, axes = plt.subplots(1, len(families), figsize=(7.16, 3.2), squeeze=False)
    tradeoff_n = config["tradeoff_qubits"] or config["max_qubits"]
    for index, family in enumerate(families):
        left = axes[0, index]
        right = left.twinx()
        reduction, errors, invalid = [], [], []
        for epsilon in epsilons:
            matched = [r for r in runs if r["family"] == family and r["n_qubits"] == tradeoff_n
                       and r["method"] == "approx_dd" and r["epsilon"] == epsilon
                       and r["status"] == "ok" and "accuracy" in r
                       and r.get("final_state_node_reduction") is not None]
            reduction.append(100 * statistics.median([r["final_state_node_reduction"] for r in matched]) if matched else np.nan)
            errors.append(statistics.median([r["accuracy"]["projected_infidelity"] for r in matched]) if matched else np.nan)
            invalid.append(any(not r["accuracy"]["is_physical"] for r in matched))
        left.plot(epsilons, reduction, "o-", color="#0072B2", label="Node reduction")
        right.plot(epsilons, errors, "s--", color="#D55E00", label="Projected 1-F")
        for epsilon, error, is_invalid in zip(epsilons, errors, invalid):
            if is_invalid:
                right.plot(epsilon, error, "x", color="black", markersize=7, markeredgewidth=1.2)
        left.set_xscale("log")
        finite_errors = [value for value in errors if math.isfinite(value)]
        if finite_errors and min(finite_errors) > 0:
            right.set_yscale("log")
            right.set_ylim(min(finite_errors) / 1.5, max(finite_errors) * 1.5)
            right.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=8))
            right.yaxis.set_major_formatter(LogFormatterSciNotation(
                base=10, labelOnlyBase=False, minor_thresholds=(float("inf"), float("inf"))))
        else:
            right.set_yscale("symlog", linthresh=1e-8, linscale=0.5)
            right.set_ylim(0, max(max(finite_errors, default=0) * 1.5, 1e-8))
        left.set_title(f"{FAMILY_LABELS[family]} (n={tradeoff_n})")
        left.set_xlabel("Approximation epsilon")
        left.tick_params(axis="y", colors="#0072B2")
        right.yaxis.set_minor_formatter(NullFormatter())
        right.tick_params(axis="y", which="both", colors="#D55E00")
        left.grid(True, alpha=0.2, linewidth=0.5)
        if index == 0:
            left.set_ylabel("Final state node reduction (%)", color="#0072B2")
        if index == len(families) - 1:
            right.set_ylabel("PSD-projected infidelity", color="#D55E00")
        if not any(math.isfinite(value) for value in errors):
            left.text(0.5, 0.5, "No matched completed runs", transform=left.transAxes,
                      ha="center", va="center", fontsize=7, wrap=True)
    figure.text(0.5, 0.97, "Blue: node reduction vs exact DD; orange: projected infidelity", ha="center", va="top", fontsize=8)
    figure.text(0.5, 0.015, "x: raw state invalid. PSD/trace-one projection is diagnostic only; raw fidelity is null in JSON.", ha="center", fontsize=7)
    figure.tight_layout(rect=(0, 0.055, 1, 0.91))
    for extension in ("png", "pdf"):
        figure.savefig(os.path.join(output_dir, "fidelity_tradeoff." + extension), dpi=600)
    plt.close(figure)


def _source_hashes() -> dict:
    result = {}
    for module in (dd_core, dd_ops, crosstalk):
        with open(module.__file__, "rb") as stream:
            result[os.path.basename(module.__file__)] = hashlib.sha256(stream.read()).hexdigest()
    with open(__file__, "rb") as stream:
        result[os.path.basename(__file__)] = hashlib.sha256(stream.read()).hexdigest()
    return result


def run_benchmarks(config: dict) -> dict:
    output_dir = os.path.abspath(config["output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "results.json")
    payload = {"schema_version": 1, "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "config": config, "environment": {"python": sys.version, "platform": platform.platform(),
               "numpy": np.__version__, "qiskit": qiskit.__version__, "matplotlib": matplotlib.__version__,
               "cpu_count": os.cpu_count(), "source_sha256": _source_hashes()},
               "definitions": {
                   "exact_dd": "Floating-point DD with epsilon=1e-15, not symbolic exact arithmetic.",
                   "active_nodes": "UniqueTable retained nonterminal nodes. No garbage collection: peak equals final retained count.",
                   "state_nodes": "Nodes reachable from the density root. Peak sampled at instruction boundaries only.",
                   "timing": "perf_counter simulation plus local gate parsing/cache/statistics; routing, process startup, dense export and fidelity excluded. No warmup; fresh process per repetition.",
                   "dense_baseline": "Complex128 dense tensor contraction and partial-trace depolarization, identical routed instruction order and ZZ pairs.",
                   "fidelity": "Uhlmann squared fidelity. Raw value is null for a nonphysical state at configured tolerance.",
                   "projection": "Hermitian part followed by eigenvalue simplex projection (PSD, trace one), used only for separately labelled fidelity diagnostics.",
                   "qft_input": "H-T-H on qubit 0 before the full QFT including final swaps.",
                   "figures": "IEEE two-column width 7.16 inches, 600 dpi PNG plus embedded-font vector PDF.",
                   "resource_limits": "Limits are per run; timeout includes worker startup. Missing runs never receive fabricated node/fidelity values."},
               "cases": [], "runs": []}
    save_json(path, payload)
    for family in config["families"]:
        for n in range(config["min_qubits"], config["max_qubits"] + 1):
            for seed in config["seeds"]:
                topology = make_topology(n, config)
                logical = generate_circuit(family, n, seed, config["random_depth"])
                routing_start = time.perf_counter()
                circuit = route_circuit(logical, topology, seed)
                case_id = f"{family}_n{n}_seed{seed}"
                instructions = [{"name": item.operation.name,
                                 "qubits": [circuit.find_bit(q).index for q in item.qubits],
                                 "params": [float(p) for p in item.operation.params]} for item in circuit.data]
                payload["cases"].append({"case_id": case_id, "family": family, "n_qubits": n, "seed": seed,
                                         "hardware": topology.to_dict(), "logical_gate_count": len(logical.data),
                                         "routed_gate_count": len(circuit.data), "routed_depth": circuit.depth(),
                                         "routing_seconds": time.perf_counter() - routing_start,
                                         "instructions": instructions})
                reference = None
                methods = [("dense", None), ("exact_dd", EXACT_EPSILON)]
                methods.extend(("approx_dd", epsilon) for epsilon in config["epsilons"])
                for method, epsilon in methods:
                    for repeat in range(1, config["repeats"] + 1):
                        print(f"[{case_id}] {method} epsilon={epsilon} repeat={repeat}", flush=True)
                        row = {"case_id": case_id, "family": family, "n_qubits": n, "seed": seed,
                               "method": method, "epsilon": epsilon, "repeat": repeat}
                        result = run_isolated(circuit, topology, method, epsilon, config)
                        matrix = result.pop("matrix", None)
                        row.update(result)
                        if row["status"] == "ok":
                            if method == "dense" and reference is None:
                                reference = matrix.copy()
                            if reference is not None:
                                metric_start = time.perf_counter()
                                with threadpool_limits(limits=config["blas_threads"]):
                                    row["accuracy"] = evaluate_fidelity(reference, matrix, config["physical_tolerance"])
                                row["metric_seconds"] = time.perf_counter() - metric_start
                            else:
                                row["accuracy_unavailable_reason"] = "dense reference did not complete"
                        print(f"  -> {row['status']}" + (f", {row['simulation_seconds']:.3f} s" if "simulation_seconds" in row else ""), flush=True)
                        payload["runs"].append(row)
                        add_compression_metrics(payload["runs"])
                        save_json(path, payload)
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["summary"] = {"total_runs": len(payload["runs"]),
                          "completed_runs": sum(r["status"] == "ok" for r in payload["runs"]),
                          "raw_nonphysical_runs": sum(not r["accuracy"]["is_physical"] for r in payload["runs"] if "accuracy" in r)}
    table = comparison_table(payload["runs"], config["table_format"] == "latex")
    print("\n" + table)
    for latex, name in ((False, "comparison.md"), (True, "comparison.tex")):
        with open(os.path.join(output_dir, name), "w", encoding="utf-8") as stream:
            stream.write(comparison_table(payload["runs"], latex) + "\n")
    plot_results(payload, output_dir)
    payload["artifacts"] = ["results.json", "comparison.md", "comparison.tex", "node_compression.png",
                            "node_compression.pdf", "fidelity_tradeoff.png", "fidelity_tradeoff.pdf"]
    save_json(path, payload)
    return payload


def parse_arguments(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--min-qubits", type=int, default=3)
    parser.add_argument("--max-qubits", type=int, default=8)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--epsilons", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--random-depth", type=int, default=3)
    parser.add_argument("--topology", choices=("linear", "ring", "complete"), default="linear")
    parser.add_argument("--p-depol", type=float, default=0.001)
    parser.add_argument("--p-cx-depol", type=float, default=0.01)
    parser.add_argument("--zz-theta", type=float, default=0.02)
    parser.add_argument("--physical-tolerance", type=float, default=1e-8)
    parser.add_argument("--timeout-seconds", type=float, default=60)
    parser.add_argument("--max-dd-nodes", type=int, default=100000)
    parser.add_argument("--max-cache-entries", type=int, default=250000)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--tradeoff-qubits", type=int, default=None)
    parser.add_argument("--output-dir", default="./benchmark_results")
    parser.add_argument("--table-format", choices=("markdown", "latex"), default="markdown")
    config = vars(parser.parse_args(argv))
    if not 1 <= config["min_qubits"] <= config["max_qubits"]:
        parser.error("require 1 <= min-qubits <= max-qubits")
    if config["topology"] == "ring" and config["min_qubits"] < 3:
        parser.error("ring topology requires at least three qubits")
    for name in ("repeats", "random_depth", "max_dd_nodes", "max_cache_entries", "blas_threads"):
        if config[name] < 1:
            parser.error(name + " must be positive")
    for name in ("timeout_seconds", "physical_tolerance"):
        if not math.isfinite(config[name]) or config[name] <= 0:
            parser.error(name + " must be finite and positive")
    if not all(math.isfinite(e) and EXACT_EPSILON < e < 1 for e in config["epsilons"]):
        parser.error("approximate epsilons must be finite and between 1e-15 and 1")
    if any(seed < 0 or seed >= 2 ** 32 for seed in config["seeds"]):
        parser.error("seeds must be in [0, 2**32)")
    config["epsilons"] = sorted(set(config["epsilons"]))
    config["seeds"] = list(dict.fromkeys(config["seeds"]))
    config["families"] = list(dict.fromkeys(config["families"]))
    if config["tradeoff_qubits"] is not None and not config["min_qubits"] <= config["tradeoff_qubits"] <= config["max_qubits"]:
        parser.error("tradeoff-qubits must be within the benchmark range")
    try:
        make_topology(config["min_qubits"], config)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    return config


if __name__ == "__main__":
    mp.freeze_support()
    run_benchmarks(parse_arguments())
