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

## Phase four: automated benchmarks

```bash
# Default sweep: GHZ, QFT and random Clifford+T, n=3 through 8.
python run_benchmarks.py

# A short check of all families.
python run_benchmarks.py --max-qubits 4 --tradeoff-qubits 4

# Multiple circuit seeds and independent timing repetitions.
python run_benchmarks.py --max-qubits 5 --seeds 2026 2027 2028 --repeats 3

# LaTeX table on stdout; both Markdown and LaTeX are always saved.
python run_benchmarks.py --max-qubits 4 --table-format latex

python -m unittest -v test_crosstalk_model test_run_benchmarks
```

The default output folder is `./benchmark_results`. Each completed or failed
run is atomically checkpointed into `results.json`. The folder also contains
`comparison.md`, `comparison.tex`, `node_compression.png`,
`fidelity_tradeoff.png`, and vector PDF versions of both figures. PNG output
is 600 dpi at 7.16-inch two-column width; PDFs embed TrueType fonts. These
settings follow the [IEEE graphics size and resolution guidance](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/resolution-and-size/).
Check the target publication's specific requirements before submission.

The three method groups use exactly the same routed instruction sequence,
depolarizing strengths and per-instruction crosstalk events:

1. **Dense baseline:** complex128 density matrix, local tensor contractions
   and independent partial-trace replacement for depolarization.
2. **Exact DD baseline:** the existing engine at epsilon `1e-15`. This is a
   near-exact floating-point baseline, not epsilon zero or symbolic arithmetic.
3. **Approximate DD:** epsilon `1e-4`, `1e-3`, and `1e-2` by default.

QFT includes final swaps and an explicit H-T-H input preparation on qubit 0.
Random Clifford+T uses deterministic seeded single-qubit layers with CX
matchings. All circuits are routed to the selected linear, ring or complete
hardware graph with fixed transpiler seed and optimization level zero.
Gate counts and the complete routed instruction list are recorded in JSON.

### Reading the metrics

- `simulation_seconds` uses `time.perf_counter()`. It includes simulator
  construction, local gate parsing, cache work and simulator diagnostics.
  It excludes routing, process startup, dense DD export, fidelity and plotting.
  Export, metric and worker wall times are recorded separately. Each repetition
  uses a fresh process with no warmup. BLAS thread count is configurable.
- `peak_active_nodes` and `final_active_nodes` count all nonterminal nodes
  retained by the unique table. They are equal because the current table has
  no garbage collection. This includes intermediate operators and states.
- `final_state_nodes` counts nodes reachable from the final density root.
  `peak_state_nodes_at_instruction_boundaries` is explicitly sampled only
  after complete instructions and includes the initial state. It is not an
  internal-operation peak. Compute-cache entry count is recorded separately.
- The first figure shows **peak retained nodes**. The tradeoff figure uses
  **final-state node reduction** against the matching exact-DD run. Missing
  baselines produce missing compression values, never invented ratios.
- `fidelity` is the squared Uhlmann fidelity requested in the experiment.
  If trace, Hermiticity or positivity fails the configured physical tolerance,
  raw fidelity and infidelity are `null`. The raw trace, minimum eigenvalue,
  negative eigenvalue mass and matrix errors remain available.
- `projected_fidelity` is a separate diagnostic: symmetrize the approximate
  matrix, project its eigenvalues onto the probability simplex, then calculate
  fidelity. The correction magnitude is recorded. This projection is never
  fed back into the DD simulation. The tradeoff plot labels this quantity and
  marks invalid raw states with crosses. It does not claim the original DD
  output was a valid density matrix.

Default safety budgets are 60 seconds, 100,000 retained nodes and 250,000
compute-cache entries per run. Use `--timeout-seconds`, `--max-dd-nodes` and
`--max-cache-entries` to change them. Timeouts and resource failures are stored
with their status and excluded from successful-run medians. A sweep may thus
finish with incomplete individual experiments. Dense matrices also grow as
`4**n`; increasing the range requires appropriate hardware.

The committed example sweep uses all three families for n=3 through 8,
one seed and one timing repetition. Its exact command is:

```bash
python run_benchmarks.py --max-qubits 8 --timeout-seconds 12 \
  --max-dd-nodes 50000 --max-cache-entries 150000 --tradeoff-qubits 4
```

The four-qubit tradeoff slice is explicit so completed methods can be compared
even when larger exact-DD runs exceed the budget. Single timing observations
are demonstration data, not statistically established performance claims.
Results may show limited compression, slower DD execution, or nonphysical
approximate states. The benchmark reports these outcomes as measured.
