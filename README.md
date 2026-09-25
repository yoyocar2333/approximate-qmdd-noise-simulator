# Approximate QMDD noise simulator

Python/NumPy matrix decision diagrams with Qiskit circuit parsing and a
topology-aware, per-instruction ZZ crosstalk model.

## Install and run

Use Python 3.10 or newer. Tested with Python 3.12, NumPy 2.3.5 and Qiskit 2.5.2.

```bash
python -m pip install -r requirements.txt
python dd_core.py
python dd_ops.py
python crosstalk_model.py
python -m unittest -v test_crosstalk_model
```

## Example

```python
from qiskit import QuantumCircuit
from crosstalk_model import CrosstalkDDSimulator, HardwareTopology, dd_trace
from dd_core import dd_to_matrix

circuit = QuantumCircuit(3)
circuit.h(0)
circuit.cx(0, 1)
circuit.ry(0.37, 2)

hardware = HardwareTopology.linear(
    3, p_depol=0.003, p_cx_depol=0.02, zz_theta=0.04
)
simulator = CrosstalkDDSimulator(circuit, hardware, epsilon=1e-12)
rho_dd = simulator.run()
print(dd_trace(rho_dd))
print(simulator.history)

# Optional dense export for a small-system comparison, outside simulation.
rho_numpy = dd_to_matrix(rho_dd, circuit.num_qubits)
hardware.to_json("hardware.json")
simulator.save_history("simulation_history.json")
```

Each `run()` starts again from the all-zero basis density with fresh tables.
Topology constructors also include `ring(n)`, `heavy_hex(distance=3)`,
`from_coupling_map(qiskit_coupling_map)` and `from_json(path)`.
`HardwareTopology(n, coupling_map)` accepts an edge list or adjacency mapping.
Adjacency is symmetrized, deduplicated and stored as an immutable mapping.
Isolated vertices are retained. Heavy-hex distance 3 produces 19 vertices.

## Model definition

For each gate in `circuit.data`, in its original order:

1. Apply the ideal unitary on its physical target qubits.
2. Apply a depolarizing channel on the gate subsystem.
3. Apply `exp(-i * zz_theta * Z tensor Z)` once for every undirected hardware
   edge incident to any target. An idle neighbor is included. An edge between
   two active qubits, including the gate's own coupling, is included once.

Separate instructions each trigger their own noise event. This convention
does not model parallel pulse overlap, gate duration, T1/T2, or calibrated
device-specific crosstalk. `zz_theta` is in radians per instruction; the
equivalent Qiskit gate is `rzz(2 * zz_theta)`.

The parameter `p` follows `D_p(rho) = (1-p)rho + p I/d` on an isolated gate
subsystem of dimension `d`. On an entangled system, the replacement term is
the maximally mixed target subsystem tensored with the spectators' reduced
state. It is not the total probability of a nonidentity Pauli error.

- One-qubit Kraus operators: `sqrt(1-3p/4) I` and `sqrt(p)/2` times X, Y, Z.
- Two-qubit Kraus operators: `sqrt(1-15p/16) II` and `sqrt(p)/4` times the
  other 15 Pauli products. All two-qubit gates use `p_cx_depol`.
- `p_depol` and `p_cx_depol` must be in `[0, 1]`; `zz_theta` can have either sign.

The previous `dd_ops.py` standalone demonstration uses the alternative Pauli
error-probability parameterization. Its Pauli probability is `3p/4` for this
module's single-qubit depolarizing strength `p`.

## Ordering and supported circuits

Qiskit qubit 0 is the least significant basis bit. The DD root represents
qubit `n-1`. `embed_operator_dd(matrix, qubits, n, table)` treats `qubits[0]`
as the local least significant bit and supports arbitrary target order and
noncontiguous physical indices without building a full-system dense operator.
Only local gate matrices are obtained through Qiskit's `Operator`.

Circuit width must equal hardware width, including idle spectators. Circuit
indices are interpreted as physical indices: supply a mapped circuit. Gates
with two targets must follow a topology edge by default. The explicit option
`enforce_coupling=False` permits ideal nonlocal gates for experiments while
retaining topology-based noise. Directional hardware gate constraints are
outside this undirected model.

Bound one/two-qubit unitary gates and barriers are supported. Barriers are
no-ops. Measurement, reset, delay, classical control, initialization instructions,
unbound parameters and wider gates raise descriptive errors. Decompose larger
gates first; remove final measurements when requesting a premeasurement density.
Circuit global phase cancels in density-matrix evolution.

## Approximation and validation

`epsilon` is a local weight-pruning and quantization tolerance, not a global
fidelity guarantee. Small epsilon is appropriate for correctness comparisons;
larger epsilon trades accuracy for sharing. Approximation can perturb trace,
Hermiticity and positivity. The simulator reports trace and does not silently
normalize or project the result. The unique table retains intermediate nodes;
history distinguishes retained nodes from nodes reachable from the current root.

The test script independently computes the depolarizing reference using partial
trace replacement, checks physical-qubit ordering against Qiskit, and validates
ZZ angle, Kraus completeness, spectators, pair deduplication, trace, Hermiticity,
positivity, JSON round trips and unsupported instructions. It also runs a
19-qubit heavy-hex example without dense density conversion. This small circuit
is a structural test, not a general scalability benchmark.

Observed maximum elementwise errors on the included deterministic cases:

| Case | Epsilon | Maximum error |
| --- | ---: | ---: |
| Six-gate, three-qubit noisy circuit against dense reference | 1e-12 | 5.00e-16 |
| Three-gate, three-qubit demonstration against dense reference | 1e-4 | 6.71e-5 |

API conventions were checked against IBM's [bit-ordering guide](https://quantum.cloud.ibm.com/docs/en/guides/bit-ordering)
and [CouplingMap documentation](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.CouplingMap).
