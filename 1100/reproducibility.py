#!/usr/bin/env python3
"""
reproducibility.py
==================

Reproduces every numerical claim in two notes on Erdos problem #1100
(the squarefree extremal function g_sf(k) for coprime adjacent divisors):

  [R] "A reduction of the squarefree coprime adjacent divisor problem,
       and a conditional golden-ratio lower bound"
  [L] "A within-layer adjacency lemma"

Model.  For a weight vector a = (a_1,...,a_k) with distinct subset sums and
sigma(S) = sum_{i in S} a_i, order the 2^k subsets of [k] increasingly by sigma.
tau_perp counts adjacent pairs (S,T) with S ∩ T = empty (= coprime adjacent
divisors when a_i = log p_i).  N(k,j) counts such adjacencies within the
size-j cardinality layer.

All randomized routines are seeded; deterministic routines take no seed.
Run `python3 reproducibility.py` for the full report, or import individual
functions.  Each function's docstring names the claim it backs.

"""

from __future__ import annotations
from itertools import combinations
from math import comb, log, sqrt, pi
import numpy as np

try:
    from sympy import primerange
    _HAVE_SYMPY = True
except Exception:  # pragma: no cover
    _HAVE_SYMPY = False


# --------------------------------------------------------------------------
# core helpers
# --------------------------------------------------------------------------

def fibonacci(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def all_subset_sums(weights):
    """Return (sums, masks) numpy arrays over all 2^k subsets, via doubling."""
    k = len(weights)
    sums = np.array([0.0]); masks = np.array([0], dtype=np.int64)
    for i in range(k):
        sums = np.concatenate([sums, sums + weights[i]])
        masks = np.concatenate([masks, masks | (1 << i)])
    return sums, masks


def tau_perp(weights) -> int:
    """Number of adjacent disjoint pairs in the full sigma-order."""
    sums, masks = all_subset_sums(weights)
    o = np.argsort(sums, kind="mergesort")
    m = masks[o]
    return int(np.count_nonzero((m[:-1] & m[1:]) == 0))


def _popcount(v: int) -> int:
    return bin(int(v)).count("1")


# --------------------------------------------------------------------------
# [R] §5 — claim 1: the exact identity  (Theorem 1)
# --------------------------------------------------------------------------

def check_identity(kmax: int = 10, seed: int = 0, reps_per_k: int = 4):
    """[R, Thm 1].  tau_perp  ==  #{ K : balanced split of K is unblocked }.

    Verified in EXACT integer arithmetic (distinct subset sums guaranteed),
    for 1 <= k <= kmax.  Reproduces the claim 'checks exactly over all K for
    k <= 10, zero mismatches'.
    """
    rng = np.random.default_rng(seed)
    mismatches = 0
    rows = []
    for k in range(1, kmax + 1):
        for _ in range(reps_per_k):
            # random distinct integer weights with distinct subset sums
            while True:
                w = sorted(int(x) for x in rng.integers(1, 50_000, size=k))
                ss = [sum(w[i] for i in c)
                      for r in range(k + 1) for c in combinations(range(k), r)]
                if len(set(ss)) == len(ss):
                    break
            t = tau_perp(w)
            c = _unblocked_count_exact(w)
            rows.append((k, t, c, t == c))
            if t != c:
                mismatches += 1
    print(f"[R Thm 1] identity check, k<=:{kmax}, exact arithmetic, "
          f"{len(rows)} vectors, mismatches = {mismatches}")
    assert mismatches == 0, "IDENTITY FAILED"
    return mismatches


def _unblocked_count_exact(weights):
    """#{ K subset of [k] : no subset sum lies strictly inside the balanced
    split interval of K }.  Exact; weights are integers."""
    import bisect
    k = len(weights)
    sums = sorted(sum(weights[i] for i in c)
                  for r in range(k + 1) for c in combinations(range(k), r))
    cnt = 0
    for r in range(k + 1):
        for Kc in combinations(range(k), r):
            tot = sum(weights[i] for i in Kc)
            best = None
            for rr in range(len(Kc) + 1):           # balanced split of K
                for sub in combinations(Kc, rr):
                    a = sum(weights[i] for i in sub)
                    imb = abs(2 * a - tot)
                    if best is None or imb < best[0]:
                        best = (imb, a, tot - a)
            wa, wb = sorted((best[1], best[2]))
            lo = bisect.bisect_right(sums, wa)
            hi = bisect.bisect_left(sums, wb)
            if hi - lo == 0:                          # nothing strictly inside
                cnt += 1
    return cnt


# --------------------------------------------------------------------------
# [R] §5 — claim 2: mean-tau_perp growth and concentration
# --------------------------------------------------------------------------

def mean_tau_growth(ks=range(8, 23), seed: int = 0):
    """[R, §5 / eq (4.2)].  Mean of tau_perp over i.i.d. Uniform[0,1] weights is
    exponential with base climbing ~1.54 -> 1.57 over k=8..22, with mean ~ max.
    Returns list of (k, mean, base, max_base).
    """
    rng = np.random.default_rng(seed)
    out = []
    for k in ks:
        S = 400 if k <= 14 else (150 if k <= 17 else (50 if k <= 19 else 20))
        vals = np.array([tau_perp(rng.uniform(0, 1, k)) for _ in range(S)])
        mean = float(vals.mean()); mx = int(vals.max())
        out.append((k, mean, mean ** (1 / k), mx ** (1 / k)))
        print(f"[R §5] k={k:2d}  mean={mean:9.1f}  base={mean**(1/k):.4f}  "
              f"max_base={mx**(1/k):.4f}  (n={S})")
    return out


# --------------------------------------------------------------------------
# [R] §5 — claim 3: layer profile, Fibonacci total, peak ~ 0.276 k
# --------------------------------------------------------------------------

def _lex_layer_counts(x):
    """Per-cardinality-layer disjoint-adjacency counts in the lexicographic
    (tight-band) order induced by weights x; returns array N[j]."""
    k = len(x)
    masks = np.arange(1 << k, dtype=np.int64)
    pop = np.array([_popcount(m) for m in range(1 << k)])
    sums = np.zeros(1 << k)
    for i in range(k):
        sums[((masks >> i) & 1).astype(bool)] += x[i]
    order = np.lexsort((sums, pop))           # primary key = cardinality
    m = masks[order]; p = pop[order]
    N = np.zeros(k + 1)
    for a in range(len(m) - 1):
        if (m[a] & m[a + 1]) == 0 and p[a] == p[a + 1]:
            N[p[a]] += 1
    return N


def layer_profile(k: int = 16, reps: int = 60, seed: int = 0):
    """[R, §5 + eq (4.1),(4.2)].  E[N(k,j)] ~ binom(k-j,j); peak at j≈0.276k;
    middle layer j=k/2 is O(1); sum_j N(k,j) ~ F_{k+1}.
    Returns (per_layer_mean, binoms, total, F_{k+1}).
    """
    rng = np.random.default_rng(seed)
    N = np.zeros(k + 1)
    for _ in range(reps):
        N += _lex_layer_counts(rng.uniform(0, 1, k))
    N /= reps
    binoms = np.array([comb(k - j, j) if 2 * j <= k else 0 for j in range(k + 1)])
    total = N.sum(); Fk = fibonacci(k + 1)
    peak = int(np.argmax(N[: k // 2 + 1]))
    print(f"[R §5] layer profile k={k} (reps={reps})")
    print(f"   peak layer j={peak} (predicted ~{0.276*k:.1f}); "
          f"middle layer N(k,{k//2})={N[k//2]:.2f}")
    for j in range(1, k // 2 + 1):
        print(f"     j={j:2d}: N={N[j]:8.1f}   binom(k-j,j)={binoms[j]:6d}   "
              f"ratio={N[j]/binoms[j]:.3f}")
    print(f"   sum_j N = {total:.1f}   F_{{{k+1}}} = {Fk}   "
          f"ratio={total/Fk:.3f}   base={total**(1/k):.4f}  (phi={(1+5**.5)/2:.4f})")
    return N, binoms, total, Fk


# --------------------------------------------------------------------------
# [R] §5 / Remark 1 — claim 4: the floor.  max tau_perp exceeds F_{k+1}.
# --------------------------------------------------------------------------

def floor_search(k: int = 12, trials: int = 6000, seed: int = 0):
    """[R, Remark 1].  The true maximum exceeds F_{k+1}: searching prime sets
    and random real weights finds tau_perp >= 293 at k=12 vs F_13 = 233.
    Returns (best_found, F_{k+1}).
    """
    rng = np.random.default_rng(seed)
    primes = (list(primerange(2, 2000)) if _HAVE_SYMPY
              else _first_primes(400))
    logs = [log(p) for p in primes]
    best = 0
    best = max(best, tau_perp(logs[:k]))                       # first k primes
    pool = list(range(min(60, len(primes))))
    for _ in range(trials):
        idx = sorted(rng.choice(pool, size=k, replace=False))
        best = max(best, tau_perp([logs[i] for i in idx]))     # spread primes
    for _ in range(trials):
        best = max(best, tau_perp(sorted(rng.uniform(1, 5, k))))  # real weights
    Fk = fibonacci(k + 1)
    print(f"[R Rmk 1] k={k}: best tau_perp found = {best}   F_{{{k+1}}} = {Fk}   "
          f"exceeds F = {best > Fk}")
    return best, Fk


def _first_primes(n):
    out, c = [], 2
    while len(out) < n:
        if all(c % p for p in out if p * p <= c):
            out.append(c)
        c += 1
    return out


# --------------------------------------------------------------------------
# [L] Prop 1(i) — claim 5: f_Delta(0) = ||f_{IH_j}||_2^2 -> sqrt(3/pi)/sqrt(j)
# --------------------------------------------------------------------------

def f_delta_at_zero(js=(2, 3, 4, 6, 8, 12, 16), grid: int = 1024):
    """[L, Prop 1(i)].  f_Delta(0) equals ||f_{IH_j}||_2^2 (Irwin-Hall L2 norm)
    and -> sqrt(3/pi)/sqrt(j).  Deterministic.  Returns list of (j, value, approx).
    """
    h = 1.0 / grid
    base = np.ones(grid) * h                       # U[0,1] mass on a fine grid
    out = []
    for j in js:
        pmf = base.copy()
        for _ in range(j - 1):
            pmf = np.convolve(pmf, base)
        val = (pmf ** 2).sum() / h                  # ||f||_2^2 = sum (pmf/h)^2 h
        approx = sqrt(3 / pi) / sqrt(j)
        out.append((j, val, approx))
        print(f"[L Prop1(i)] j={j:2d}  f_Delta(0)={val:.5f}  "
              f"sqrt(3/pi)/sqrt(j)={approx:.5f}  ratio={val/approx:.4f}")
    return out


# --------------------------------------------------------------------------
# [L] §3 — claim 6: swap-blockers are negligible; empty-given-close is Theta(1)
# --------------------------------------------------------------------------

def _layer_sorted(k, j, rng):
    idx = list(combinations(range(k), j))
    masks = np.array([sum(1 << i for i in c) for c in idx], dtype=np.int64)
    x = rng.uniform(0, 1, k)
    sums = np.array([x[list(c)].sum() for c in idx])
    o = np.argsort(sums, kind="mergesort")
    return masks[o], sums[o]


def swap_diagnostics(ks=(14, 16, 18, 20, 22), seed: int = 0):
    """[L, §3].  At j≈0.276k: (i) nearest-in-sum neighbor overlap ≈ j^2/k, not
    j-1 (swaps don't control local spacing); (ii) slope C = E[blockers]/(gap in
    spacings) is bounded (~0.55); (iii) swap-type blocker fraction -> 0.
    Returns list of dict rows.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for k in ks:
        j = max(2, round(0.276 * k))
        m, s = _layer_sorted(k, j, rng); M = len(m)
        nn_overlap = np.mean([_popcount(m[i] & m[i + 1]) for i in range(M - 1)])
        lo, hi = int(0.45 * M), int(0.55 * M)
        sp = float(np.median(np.diff(s[lo:hi])))
        thr = 3 * sp
        tot_blk = tot_gap = 0.0
        swap = total = 0
        for p in range(M):
            q = p + 1
            while q < M and s[q] - s[p] < thr:
                if (m[p] & m[q]) == 0:
                    tot_blk += (q - p - 1); tot_gap += (s[q] - s[p]) / sp
                    for r in range(p + 1, q):
                        total += 1
                        if (_popcount(m[p] & m[r]) >= j - 1 or
                                _popcount(m[q] & m[r]) >= j - 1):
                            swap += 1
                    break
                q += 1
        C = tot_blk / tot_gap if tot_gap else float("nan")
        frac = swap / total if total else 0.0
        rows.append(dict(k=k, j=j, nn_overlap=nn_overlap, generic=j * j / k,
                         swap_value=j - 1, C=C, swap_frac=frac))
        print(f"[L §3] k={k:2d} j={j}: nn_overlap={nn_overlap:.2f} "
              f"(generic j^2/k={j*j/k:.2f}, swap j-1={j-1})  "
              f"C={C:.3f}  swap_frac={frac:.4f}")
    return rows


# --------------------------------------------------------------------------
# [L] §4(a)(iii) — claim 7: the hypergeometric concentration (UNPROVED step)
# --------------------------------------------------------------------------

def hypergeometric_concentration(kjs=((18, 5), (20, 6), (22, 6), (26, 7))):
    """[L, §4(a)(iii)] — the one analytically UNPROVED step.
    Checks   sum_{m>=1} C(2j,j-m) C(k-2j,m) / sqrt(m)  <=  C * C(k,j)/sqrt(j),
    reporting the empirical C (~1.5 near the peak), and the decay of the
    internal (m=0) fraction C(2j,j)/C(k,j) -> 0.  Deterministic.
    """
    rows = []
    for k, j in kjs:
        tot = sum(comb(2 * j, j - m) * comb(k - 2 * j, m) / sqrt(m)
                  for m in range(1, j + 1))
        ref = comb(k, j) / sqrt(j)
        internal = comb(2 * j, j) / comb(k, j)
        rows.append((k, j, tot / ref, internal))
        print(f"[L §4a] k={k} j={j}: C = sum/(C(k,j)/sqrt j) = {tot/ref:.3f}   "
              f"internal m=0 fraction = {internal:.4f}")
    return rows


# --------------------------------------------------------------------------
# full report
# --------------------------------------------------------------------------

def main():
    line = "=" * 72
    print(line); print("REDUCTION NOTE [R]"); print(line)
    check_identity(kmax=10)
    print()
    mean_tau_growth(ks=range(8, 19))      # trim top of range for speed
    print()
    layer_profile(k=16, reps=40)
    print()
    floor_search(k=12, trials=3000)
    print()
    print(line); print("WITHIN-LAYER LEMMA [L]"); print(line)
    f_delta_at_zero()
    print()
    swap_diagnostics(ks=(16, 18, 20, 22))
    print()
    hypergeometric_concentration()


if __name__ == "__main__":
    main()
