# Method

## Circuit and cost
* **Ansatz.** L layers on n qubits (qubit 0 is the most significant bit). Each layer applies a rotation
  RY(theta) on every qubit, controlled-Z (CZ) gates on even bonds, CZ gates on odd bonds, then
  single-qubit depolarising noise of Pauli error probability p on every qubit. All gates are real, so
  the exact density matrix stays real and is simulated as a batched float64 tensor (PyTorch, on a
  graphics processing unit (GPU) when available).
* **Depth schedules (ablation).** *log*: L = ceil(log2 n) + 1, the regime of Cerezo et al. (2021) in
  which a local cost is trainable without noise; *linear*: L = n, the regime of Wang et al. (2021).
* **Costs.** Starting from |0...0>: global cost C_G = 1 - P(0...0); local cost
  C_L = 1 - (1/n) sum_j P(qubit j = 0). Both come from the same measured bit strings.
* **Noise sweep.** p in {0, 0.005, 0.01, 0.02} per qubit per layer; the channel is
  rho -> (1 - 4p/3) rho + (4p/3) tr_q(rho) (x) I/2.

## Experiment 1: gradient statistics (scaling study)
For n = 3 to 8, three seeds and 256 random initialisations per seed (angles uniform in [0, 2 pi), one
random parameter each), compute the exact parameter-shift gradient (C(+pi/2) - C(-pi/2))/2. Record the
variance, and the number of shots per circuit for which the shot-noise standard deviation of the
gradient estimate equals the root-mean-square gradient ("unit signal-to-noise ratio (SNR) shots").
Fit log-log slopes against d = 2^n per seed.

## Experiment 2: shot-limited training
n in {4, 6, 8}, log-depth schedule, p in {0, 0.01}, 5 seeds, 200 shots per circuit evaluation, and
equal total shots per run (25 Adam steps x 2 x n x L circuit evaluations). Methods:
* **Standard quantum baseline:** Adam with parameter-shift gradients on the global cost, as in the
  barren-plateau literature (Cerezo et al., 2021).
* **Variant under test:** Adam with the local cost.
* **Classical optimiser baselines:** simultaneous-perturbation stochastic approximation (SPSA; two
  cost evaluations per step) on either cost; random search (best of the same number of random
  parameter sets, chosen by measured cost) as the null; the untrained random initialisation as reference.
* Hyper-parameters (Adam learning rate 0.1; SPSA gain 3.0 and perturbation 0.2) were set on held-out
  seeds (100 to 102) at n = 6, p = 0.01, not on the reported seeds.
* **Metric:** final noise-free global infidelity 1 - |<0|psi(theta)>|^2 of the trained parameters, so
  both costs are scored on the same ideal quantity.

## Statistics and reproducibility
Mean and standard deviation over seeds are reported; per-seed rows are in `results/`. Torch
generators are seeded from the (seed, n) pair. Quick mode (`--quick`) shrinks the grid to n = 3 to 5
with 2 seeds and writes to a temporary directory.

## Limits of the model
Noise is layer-wise single-qubit depolarising, not gate-resolved or correlated; readout error and
coherent error are absent; the target state is |0...0>, a favourable case for a local cost.
