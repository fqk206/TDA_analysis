# TDA analysis — findings

A record of what we learned while building and testing the simplified CRLI /
Laplace inversion workflow on `data/25ugml_ivALPS_DLS.txt` (iv-ALPS, 25 µg/mL).

## The sample at a glance

| Quantity | Value |
|---|---|
| Main colloid (median Rh) | ~23.3 nm |
| Small species ("free fluorophore") | real, recovered at **~0.75–0.8 nm** |
| Small-species fraction | ~3% of the area |
| Fit residual | ~1.9× the estimated noise (structured, apex-concentrated) |

## The question

The CRLI fit kept producing a small low-Rh mode (~3 nm) carrying little mass.
The whole investigation was one question: **is that small mode a numerical
artefact of the ill-posed inversion, or a real feature of the sample?**

## What we found, step by step

### 1. It is not an edge/boundary artefact, but boundaries matter
The mode is *interior* (the distribution is zero at the grid's lower edge), so a
zero-at-boundary fix does nothing for it. However, **where you put the grid edge
matters a lot**: truncating the grid *at* the mode (e.g. an auto-narrowed ~3 nm
floor) makes it worse — a grid endpoint is penalised on one side only, so mass
piles up against it. Extending the grid *down* (to 1.5 nm, then 0.3 nm) gives the
mode room to taper to zero on both sides, and the smoothness penalty can then
tame it. **Lesson: never truncate the grid through a feature you care about.**

### 2. Regularization (a ridge penalty) does *not* remove it
We added a ridge (amplitude) penalty `β = ρ·α` on top of the second-difference
smoothness. Pipeline-accurate testing showed it fails for this problem:

- A **uniform ridge** suppresses the mode only by **broadening the main peak**
  (p10–p90 went 32 → 68+ nm) and tripling the residual — an L2 norm is minimised
  by spreading mass out.
- Tied to the α-selector, the ridge just makes the selector pick a smaller α and
  cancels itself out.

It is left in `fit_crli_simple.py` but **off by default** (`RIDGE_RHO = 0.0`).
The reason it can't help: the residual is already ~1.9× noise and **structured at
the peak apex** (a sign-alternating misfit growing toward the tip). The mode is
*load-bearing* — it fits real signal, not noise — so any penalty strong enough to
delete it must fight real structure.

### 3. The truncation test: it is a real feature
We zero the colloid mass below a cutoff (4 nm) and ask how much signal that mass
was contributing, compared to the noise:

| Quantity | Value |
|---|---|
| Mass below 4 nm | ~1.4% of colloid area |
| **Peak signal from that mass** | **~28× noise** (≈4% of peak height) |
| RMS signal from that mass | ~8× noise |

The contribution is **far above noise → a real feature**, not a numerical ghost.
(Implemented as the truncation-test plot in `fit_crli_simple.py`.)

### 4. Why a tiny mass matters: additive signals
The Taylor kernel is **tall and narrow for small Rh, short and broad for large
Rh**. So a small mass fraction of a small species contributes disproportionately
at the apex. The forward simulator (`simulate_two_gaussian.py`) makes this
concrete: a species at **3% of the mass** contributes **~14–16% of the peak
signal** (≈16% at 0.59 nm, ≈14% at 0.80 nm). This is the additive-signal
confusion that makes a small free-fluorophore-like population easy to mistake for
an artefact.

### 5. The size is the free fluorophore running slightly large
"Real signal" is not the same as "a distinct 3 nm species." A 0.59 nm free
fluorophore and a 1–4 nm colloid band are **degenerate** at the apex, so the
inversion splits the small-species signal between them in a constraint-dependent
way (relaxing the free fraction shifts mass between the two).

The free-size scan (`diagnose_free_size.py`) refits over assumed free sizes:

| free Rh | overall/noise | apex/noise | leftover sub-4 nm signal |
|---|---|---|---|
| 0.59 nm | 1.45 | 1.83 | 8.4× noise |
| **0.80 nm** | **1.51** | **1.79** | **0× noise** |
| ≥1.0 nm | 2.4–4.7 | 4.4–9.5 | (fit degrades badly) |

At **~0.8 nm the small colloid mode vanishes with no loss of fit quality**;
sizes ≥1 nm fail. So the data want the dye at ~0.7–0.8 nm — modestly above its
0.59 nm nominal value, **not** a 3 nm population. The "3 nm ghost" was the
inversion nudging the effective free size up from a too-small fixed 0.59 nm.

### 6. The pure Laplace inversion confirms it independently
`fit_laplace.py` drops the free-fluorophore term entirely and inverts a single
distribution over a grid starting near zero (0.2–0.3 nm). It **recovers the small
species on its own at a peak of ~0.75 nm**, cleanly separated from the colloid by
a zero gap, tapering to zero at the low edge — and with a *slightly better*
residual (0.00203) than the explicit-free-term fit, precisely because it can
place the small species at its natural size instead of pinning it at 0.59 nm.

## Conclusion

The small low-Rh mode is **a real feature, not an artefact**: it is the free
fluorophore appearing at **~0.75–0.8 nm**, slightly larger than its 0.59 nm
nominal Rh. Three independent methods agree:

| Method | small-species size |
|---|---|
| Free-size scan | best at ~0.8 nm |
| Two-Gaussian forward model | 0.59 vs 0.80 differ ~2% of peak signal |
| Pure Laplace inversion | emergent peak at ~0.75 nm |

Practical guidance for this sample: either set `FREE_RH_NM ≈ 0.75` in the CRLI
fitter, or use the pure Laplace inversion, which finds it automatically. Ideally
confirm the dye's apparent size with a dye-only/blank run.

## Scripts produced

| Script | Purpose |
|---|---|
| `fit_crli_simple.py` | CRLI fit: free fluorophore as a fixed-Rh term + smooth colloid distribution. Includes the truncation-test diagnostic. |
| `fit_laplace.py` | Pure Laplace inversion: one distribution over the full size range, no free-fluorophore term — recovers the small species itself. |
| `diagnose_free_size.py` | Refits over a range of assumed free sizes; reports misfit + leftover small-species signal to pin down the dye's apparent size. |
| `simulate_two_gaussian.py` | Forward two-Gaussian Taylorgram generator: shows how a small low-fraction species contributes to the signal. |

Shared notes:
- Background subtraction (linear edge baseline) and `t0` are estimated once in
  `fit_crli_simple.prepare`, reused by the inversion scripts.
- `PEAK_FRACTION` controls how much of the peak is fit (0.50 = rising half, 1.00
  = whole peak).
- The residual floor (~1.8× noise at the apex, independent of free size) is an
  irreducible peak-shape mismatch — small, and not associated with spurious mass.
