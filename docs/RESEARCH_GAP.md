# Research gap

## State of the art

Variational quantum algorithms (VQAs) train a parameterized circuit against a measured cost and are
the main candidate for useful work on noisy intermediate-scale quantum (NISQ) devices. Their central
obstacle is the barren plateau (BP): the gradient variance shrinks exponentially with the number of
qubits, so the number of measurement shots needed to see a descent direction explodes (Larocca et al.,
2025, doi:10.1038/s42254-025-00813-9). Two mechanisms are established.

* **Cost-function-dependent BPs.** A cost built from a *global* observable has exponentially vanishing
  gradients even for shallow layered circuits; a cost built from *local* observables has at worst a
  polynomially vanishing gradient while the depth is O(log n) (Cerezo et al., 2021,
  doi:10.1038/s41467-021-21728-w).
* **Noise-induced BPs (NIBPs).** For local Pauli noise, the gradient vanishes exponentially in n once
  the depth grows linearly in n, independently of initialisation (Wang et al., 2021,
  doi:10.1038/s41467-021-27045-6).

The Variational Quantum Eigensolver (VQE) review lists the number of circuit repetitions and the
effect of noise on trainability among the four main open directions (Tilly et al., 2022,
doi:10.1016/j.physrep.2022.08.003).

## Related work

| Paper | Year | Role | What it does | What it does NOT cover |
|---|---|---|---|---|
| Cerezo et al., doi:10.1038/s41467-021-21728-w | 2021 | landmark | Proves global-cost BPs and polynomial local-cost gradients at O(log n) depth | Hardware noise; a measured shot cost; the noisy global-versus-local comparison |
| Wang et al., doi:10.1038/s41467-021-27045-6 | 2021 | landmark | Proves NIBPs for local Pauli noise at linear depth | A quantitative comparison of how the two cost types are hit; finite-n exponents |
| Larocca et al., doi:10.1038/s42254-025-00813-9 | 2025 | review | Surveys BP causes (ansatz, observable, noise) and mitigations | Which cause dominates at a given noise level; a break-even map |
| Tilly et al., doi:10.1016/j.physrep.2022.08.003 | 2022 | review | VQE best practices; flags shots and noise as open problems | Quantitative shot-versus-noise scaling for the cost choice |
| Kandala et al., doi:10.1038/nature23879 | 2017 | landmark | Hardware-efficient ansatz on superconducting qubits | Gradient scaling with qubit number |
| Sim et al., doi:10.1002/qute.201900070 | 2019 | landmark | Expressibility and entangling-capability descriptors for parameterized circuits | Link between these descriptors, noise and gradient variance |
| Grimsley et al., doi:10.1038/s41467-019-10988-2 | 2019 | landmark | Adaptive ansatz growth for shallow circuits | Cost-locality choice; noisy gradient statistics |
| O'Malley et al., doi:10.1103/PhysRevX.6.031007 | 2016 | landmark | Early VQE experiment with evidence of error tolerance at two qubits | Scaling beyond a few qubits |
| Schuld et al., doi:10.1103/PhysRevA.101.032308 | 2020 | landmark | Circuit-centric classifiers; reports noise resilience in simulation | Trainability at growing width |
| Huang and von Lilienfeld, doi:10.1021/acs.chemrev.0c01303 | 2021 | review | Machine learning in chemical compound space | Variational circuits (adjacent field: classical surrogates) |
| Peng et al., doi:10.1016/j.optcom.2026.133879 | 2027 | adjacent | Quantum communication protocol with explicit sampling and finite-resource bounds | Variational circuits; its habit of stating assumptions explicitly is the methodological analogue here |
| Akrom et al., doi:10.1016/j.optcom.2026.133925 | 2027 | adjacent | Statistical benchmarking of post-quantum cryptography on classical hardware | Variational circuits; a reminder that a cost claim needs a measured, significance-tested baseline |

## Gap

The two theorems are proved for different mechanisms and are usually applied one at a time: choose a
local cost to escape the cost-dependent BP, then argue about noise separately. What is missing is a
*measured, finite-size* answer to a practical question: **on a noisy device, does the local-cost
advantage survive, and how many shots does it save?** A weight-n Pauli expectation decays under
single-qubit Pauli noise like (1 - 4p/3)^(nL) while a weight-one expectation decays like
(1 - 4p/3)^L, so the noise may treat the two costs differently, although the NIBP theorem is stated
for both.

**Novelty (one sentence).** This work measures, in exact density-matrix simulation up to 8 qubits, the
gradient-variance and unit-signal-to-noise shot-cost exponents of global versus local costs as a
joint function of depth schedule and noise level, and tests whether the resulting advantage survives
equal-shot-budget training against a classical optimiser baseline.

## Hypotheses and success metrics (fixed before the final runs)

Slopes are least-squares slopes of log10 gradient variance against log10 d, with d = 2^n, n = 3 to 8,
reported as the mean over three seeds.

* **H1 (locality, noise-free).** At logarithmic depth and p = 0: global slope <= -0.7 and local slope >= -0.5.
  Falsified if either bound fails.
* **H2 (locality survives noise at logarithmic depth).** At p = 0.01, local slope minus global slope >= 0.4.
* **H3 (noise treats the costs differently).** At linear depth the noise-induced slope change between
  p = 0 and p = 0.02 is more negative for the global cost than for the local cost by at least 0.1.
  This contradicts a purely cost-agnostic reading of NIBPs and is falsified if the gap is below 0.1.
* **H4 (training).** With equal total shots, Adam with the local cost reaches a lower median final
  noise-free infidelity than Adam with the global cost at n = 8, p = 0.01.
* **H5 (null).** Every trained method beats random search at the same shot budget; if not, the
  setting is beyond the trainability threshold.

Scale statement: all numbers are exact noisy simulation at n <= 8 qubits on one laptop. They are
finite-size exponents, not asymptotic statements, and are not hardware data.
