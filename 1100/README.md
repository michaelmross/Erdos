# Reproducibility — Erdős #1100 squarefree extremal function

This module reproduces every numerical claim in two companion notes to *A Reduction of the Squarefree Coprime Adjacent Divisor Problem and a Conditional Golden-Ratio Lower Bound* (M. M. Ross, June 2026, [zenodo.20769000](https://zenodo.org/records/20769000)) on the squarefree extremal function `g_sf(k)` for coprime adjacent divisors (Erdős problem #1100):

- **[R]** *A reduction of the squarefree coprime adjacent divisor problem, and a conditional golden-ratio lower bound*
- **[L]** *A within-layer adjacency lemma*

All randomized routines are **seeded**; deterministic routines take no seed. The
numbers below are what the notes cite; exact trailing decimals depend on the seed
and sample sizes, but the trends and orders are stable.

## Setup

```
python3 -m pip install numpy sympy   # sympy optional (a prime fallback is built in)
python3 reproducibility.py           # full report, a few minutes
```

Or import individual functions:

```python
from reproducibility import check_identity, layer_profile, hypergeometric_concentration
check_identity(kmax=10)
```

## Claim → function map

| Note | Claim | Function | Reproduces |
|------|-------|----------|------------|
| [R] Thm 1 | The exact identity `τ⊥ = #{unblocked balanced splits}` | `check_identity` | 0 mismatches in exact arithmetic, k ≤ 10 |
| [R] §5, eq (4.2) | Mean `τ⊥` is exponential, base climbs ≈1.55→1.57, mean ≈ max | `mean_tau_growth` | base column over k = 8…22 |
| [R] §5, eq (4.1)–(4.2) | `E[N(k,j)] ≍ C(k−j,j)`; peak j ≈ 0.276k; middle layer O(1); `Σ N ≍ F_{k+1}` | `layer_profile` | per-layer ratios, peak, total / F_{k+1} |
| [R] Remark 1 | `φ` is a floor: max `τ⊥` exceeds `F_{k+1}` | `floor_search` | τ⊥ ≥ 293 vs F₁₃ = 233 at k = 12 |
| [L] Prop 1(i) | `f_Δ(0) = ‖f_IH_j‖₂² → √(3/π)/√j` | `f_delta_at_zero` | exact, deterministic |
| [L] §3 | Swap-blockers don't control local spacing; empty-given-close is Θ(1) | `swap_diagnostics` | nn-overlap ≈ j²/k (not j−1); bounded C; swap fraction → 0 |
| [L] §4(a)(iii) | **The one unproved step**: `Σ_{m≥1} C(2j,j−m)C(k−2j,m) m^{−1/2} ≤ C·C(k,j)/√j` | `hypergeometric_concentration` | C ≈ 1.5; internal m=0 fraction → 0 |

## Notes on the model

For a weight vector `a = (a₁,…,a_k)` with distinct subset sums and
`σ(S) = Σ_{i∈S} a_i`, order the 2^k subsets of `[k]` increasingly by `σ`.
`τ⊥` counts adjacent pairs `(S,T)` with `S ∩ T = ∅` (these are the coprime
adjacent divisors when `a_i = log p_i`). `N(k,j)` counts such adjacencies inside
the size-`j` cardinality layer, in the tight-band lexicographic limit
`a_i = 1 + ε x_i`, `ε → 0`.

## Caveats (as stated in the notes)

- The identity (`check_identity`) and the layer reduction are **proved**; the
  golden-ratio lower bound is **conditional**.
- `hypergeometric_concentration` backs the single analytically **unproved** step
  of [L]; it is numerical evidence, not a proof.
- `φ` is a **floor**, not the growth rate (`floor_search` shows the max exceeds
  `F_{k+1}`), so `g_sf(k)^{1/k} > φ`.

## Reproducing the headline numbers individually

```python
import reproducibility as rep
rep.check_identity(kmax=10)                       # [R] Theorem 1
rep.mean_tau_growth(range(8, 23))                 # [R] §5  (full k range)
rep.layer_profile(k=16, reps=60)                  # [R] §5
rep.floor_search(k=12, trials=6000)               # [R] Remark 1
rep.f_delta_at_zero()                             # [L] Prop 1(i)
rep.swap_diagnostics()                            # [L] §3
rep.hypergeometric_concentration()                # [L] §4(a)(iii)
```

Seeds default to `0`; pass `seed=` to vary. Determinism makes the printed tables
byte-stable across runs on the same NumPy version.
