# Hou 6 × 10 ns experimental ranking

Default grid; 39 online replicas plus 3 whole-replica offline recoveries. Table S1 experimental dG labels; lower score and lower label both mean stronger binding.

| Group | n | Spearman rho | Kendall tau-b | Concordant pairs | Exact two-sided p (rho) |
|---|---:|---:|---:|---:|---:|
| a1_a4_same_series | 4 | 1.000 | 1.000 | 6/6 | 0.0833 |
| all_seven_exploratory | 7 | 0.750 | 0.619 | 17/21 | 0.0663 |
| a5_a7_secondary | 3 | 1.000 | 1.000 | 3/3 | 0.3333 |

| Group | Block ns | Sampling-bootstrap rho 2.5–97.5 percentiles |
|---|---:|---:|
| a1_a4_same_series | 1 | 0.800 … 1.000 |
| a1_a4_same_series | 2 | 0.800 … 1.000 |
| a1_a4_same_series | 5 | 0.800 … 1.000 |
| all_seven_exploratory | 1 | 0.571 … 0.929 |
| all_seven_exploratory | 2 | 0.571 … 0.929 |
| all_seven_exploratory | 5 | 0.571 … 0.929 |
| a5_a7_secondary | 1 | 0.500 … 1.000 |
| a5_a7_secondary | 2 | 0.500 … 1.000 |
| a5_a7_secondary | 5 | 0.500 … 1.000 |

All pairwise orders and resampling frequencies: `pairs.csv`. Candidate-composition bootstrap and its undefined-draw counts: `ranking.json`.

- Primary chemical-series comparison a1-a4; all-seven and a5-a7 comparisons are supplementary.
- Table S1 labels are used as reported; original assay types/conditions are not newly established as uniform.
- Unknown experimental errors are not assigned an invented distribution.
- Score is entropy-free MM/PBSA, not an absolute binding free-energy prediction.
- Permutation p values condition on these fixed candidate labels; small n limits interpretation.
- Sampling bootstrap conditions on visited states, with six replicas and circular contiguous time blocks.
- Bootstrap percentiles are sensitivity diagnostics; unvisited states and numerical/model errors are absent.
- Candidate bootstrap is separate from MD bootstrap; constant-candidate draws have undefined correlation and are counted.
- h0.4 has no complete 6x10-ns rescoring here and is not declared converged; no grid is chosen from correlation.
