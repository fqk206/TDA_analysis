# t0 robustness — current implementation and options

Prompted by a concrete failure: on `data/0p05perc.txt` (0.05% concentration —
very low signal-to-noise), `argmax` locked onto a noisy sample ~2.3 s from
where the real peak sits, and the recovered t0 drifted noticeably.
Retightening despiking (uncommitted change in `fit_crli_simple.py`,
`DESPIKE_WINDOW`/`DESPIKE_SIGMA` 4/18 → 8/8) happened to fix this instance,
by flagging that sample as a spike — even though, as §2's Option 1 write-up
found, it isn't really a narrow instrumental artefact so much as the single
highest fluctuation on a broad, flat, noisy plateau. This note documents how
t0 and despiking work today, why a single bad sample can corrupt t0 even when
despiking mostly works, and three ways — cheap to very expensive — to make t0
robust to this class of failure without relying on despiking alone. All costs
below are measured on this machine (single run/prototype, `0p05perc.txt`,
CRLI unless noted) — treat them as order-of-magnitude, not exact.

## 1. Current implementation

### 1.1 De-spiking — Hampel identifier (`tda/preprocessing/steps.py`)

`despike_mask` flags a sample as a spike if it deviates from a *rolling
median* of its neighbours by more than `n_sigma` robust standard deviations:

1. `local = median_filter(y, size=window)` — rolling median, `window` samples wide.
2. `resid = y - local`.
3. `scale = 1.4826 * MAD(resid)` — **one global scale for the whole trace.**
4. Flag `|resid| > n_sigma * scale`.

Flagged points are AND-ed out of `good_mask` (dropped, not smoothed or
interpolated) before baseline subtraction, noise estimation, and t0 detection
all run on what's left. `window` has to be wider than the broadest expected
spike but narrower than the (broad) Taylor peak, or the peak itself gets
flagged as an outlier against its own rolling median.

Because `scale` is a single number for the entire trace, sensitivity is
uniform everywhere. It can't be tightened specifically around the peak (where
a missed spike does the most damage to t0) without also tightening it on the
flat baseline, where the existing setting is already working fine.

Current tuning (`fit_crli_simple.py`, **uncommitted**): `DESPIKE_WINDOW=8,
DESPIKE_SIGMA=8.0`, tightened from the shared library default (`window=4,
n_sigma=18.0`, still the default in `Despike`'s dataclass) specifically to
catch the `0p05perc` spike. Measured effect on that file:

| | window=4, σ=18 (old) | window=8, σ=8 (current) |
|---|---:|---:|
| spikes flagged | 1 | 3 |
| seed t0 | 105.72 s | 103.48 s |

A ~2.2 s difference in the *seed* from despike tuning alone, before any
`T0_MODE="fit"` refinement runs.

### 1.2 t0 seed — argmax + local parabola (`tda/preprocessing/pipeline.py`, `tda/numerics.py`)

Every `PreparedTrace` carries one detected `t0_s`, computed identically
regardless of which model or `T0_MODE` later consumes it:

1. `local_peak_i = argmax(corrected_signal[good])` — the single highest surviving sample.
2. `quadratic_peak_time`: fit an **unweighted degree-2 polynomial**
   (`np.polyfit`) through the 9 points at `local_peak_i ± 4` samples, and take
   the parabola's vertex as t0 (falls back to the raw sample time if the fit
   opens upward or the vertex lands outside the window).

This *is already* "fit a parabola" — just over a fixed 9-sample window, with
ordinary (non-robust) least squares, seeded by a bare `argmax`. Two ways a
surviving spike still corrupts it even when despiking mostly works:

- the spike **is** the argmax sample → the window is centered on the wrong point entirely;
- the spike is a **neighbour** of the true peak, inside the ±4-sample window → the window is right, but the unweighted fit still gets pulled toward the outlier.

### 1.3 `T0_MODE="fit"` — bounded refinement (`tda/fitting.py`)

`refine_t0` runs `scipy.optimize.minimize_scalar(objective, bounds=(seed ±
t0_refine_bound_s), method="bounded", xatol=1e-3)` — a Brent-style bounded
1-D search, default bound ±5 s. It starts from the **same** seed as §1.2:
"fit" mode doesn't re-derive an independent t0, it only searches locally
around whatever `argmax` + parabola produced. `objective(t0)` differs per
model:

| Model | Objective per candidate t0 | Cost/eval (measured) |
|---|---|---:|
| CRLI (`_crli_scan_at_t0`) | rebuild kernel, NNLS-solve the full 80-α regularized inversion over `n_rh_grid=120`, take best RMS | ~474 ms |
| Laplace (`_invert_at_t0`) | same shape, `n_rh_grid=160` | ~769 ms |
| global-series (`_resolve_trace_t0`) | 2-point (dye + apparent colloid) linear NNLS misfit; resolved once at load time, *not* inside the outer optimizer | ~0.5 ms |

Because the search is bounded *to the seed*, it can only partially rescue a
bad seed, and only if the true t0 is within `t0_refine_bound_s`. Measured on
`0p05perc` (CRLI, rising side):

| despiking | seed t0 | `T0_MODE="fit"` result |
|---|---:|---:|
| old (w=4, σ=18) | 105.72 s | 103.17 s |
| current (w=8, σ=8) | 103.48 s | 103.01 s |

Even starting from the old, spike-corrupted seed, the bounded search still
finds its way to roughly the right neighbourhood here, because the true peak
happens to sit within the ±5 s bracket around the bad seed — but it converges
to a measurably different point (103.17 s vs. 103.01 s) than the well-seeded
run, since it's only ever polishing within whatever window the seed handed
it. Had the seed error exceeded 5 s, `T0_MODE="fit"` would have returned a
confidently wrong answer bounded by the wrong bracket, with nothing in the
output flagging that anything was off. Confirmed by direct instrumentation:
the search takes **13 objective evaluations** before converging, for both
CRLI and Laplace on this file (bounded Brent's iteration count is governed by
bracket width and `xatol`, not by which model's objective it's wrapping) —
plus one final re-solve at the chosen t0, so ~14 full inversions total.

### 1.4 Cost of the current pipeline (measured, `0p05perc.txt`)

| step | CRLI | Laplace |
|---|---:|---:|
| despike + baseline + seed t0 (`prepare()`) | ~4 ms | ~4 ms |
| `T0_MODE="auto"` (1 objective eval) | ~463 ms | ~626 ms |
| `T0_MODE="fit"` (~14 objective evals) | ~6.8 s | ~8.8 s |

**Peak/t0 localization is already essentially free (~4 ms); all the cost
lives in the nested inversion that "fit" mode wraps around it (~14-15× a
single fit).** This matters for judging the options below: the seed is not
where the compute budget goes today, so making the seed more robust is
nearly free, while making the nested search itself more thorough multiplies
an already ~14×-inflated cost.

## 2. Three ways to make t0 more robust

Ordered cheapest → most expensive. They're independent and stack.

### Option 1 — Robust, wider-window peak localization (cheap: ~1-6 ms/trace)

> **Status: implemented** (`tda/numerics.py`, `tda/preprocessing/pipeline.py`).
> The two-stage design below is what shipped; it turned out to need both
> stages — the first prototype (wide robust parabola alone, still centred on
> a raw `argmax`) was tried first and did **not** fully fix `0p05perc`, which
> is why it's documented here too.

Replacing just the 9-point unweighted OLS parabola with a wider robust fit
(Huber quadratic, ±25 samples, ~1.1 ms; a Gaussian variant was also
prototyped, ~5.7 ms) cut the seed error on `0p05perc` under the *old*
despiking from 105.72 s to ~105.75 s — essentially no improvement. The reason
came out of inspecting the raw window around that `argmax` sample: on this
file (0.05% concentration, very low signal-to-noise) there is no sharp,
narrow, isolated artefact spike at the peak. The whole ~100–106 s region is a
broad, flat, noisy plateau where a dozen samples are all within ~2% of each
other in height, and the single highest one (t=105.75 s) simply happened to
win `argmax` by chance — roughly 2.3 s (~27 samples) from where the other
samples cluster, further than a ±25-sample window can reach *even robustly*,
because the window itself is centred on the wrong sample from the start. This
is a distinct failure mode from "a spike sits inside an otherwise-good
window" (which the robust loss does fix, and which is what the earlier
Huber-vs-OLS prototype numbers above demonstrate) — here the problem is
upstream, in `argmax` choosing the window's centre in the first place.

The fix is two stages, each defending a different part of the process:

1. **Robust coarse localization.** `argmax` a short median-filtered copy of
   the signal (`scipy.ndimage.median_filter`, `PEAK_LOCATE_MEDIAN_WINDOW = 9`
   samples) instead of the raw signal, to pick `i_peak`. A median filter
   can't be won by any single sample the way `argmax` can, regardless of
   whether despiking flagged it — this runs whether or not `Despike` is even
   in the pipeline.
2. **Robust sub-sample refinement.** `quadratic_peak_time`, now a Huber-loss
   fit (`scipy.optimize.least_squares`) over `half_window=25` samples around
   `i_peak` (up from a 9-point unweighted `np.polyfit`), so a residual outlier
   *inside* that window still can't dominate the vertex estimate.

Measured on `0p05perc` after both stages, seed t0 is now **identical
regardless of despiking tuning** — old despiking (window=4, σ=18, the
setting that originally broke) and current despiking (window=8, σ=8) both
land on 103.19 s, versus the 105.72 s / 103.48 s split before this change.
`T0_MODE="auto"` alone (no nested search at all) now reaches a good answer;
`T0_MODE="fit"`'s expensive refinement is no longer load-bearing for this
file. Full pipeline cost is unchanged at ~3-6 ms (`prepare()` timing before
vs. after this change is within run-to-run noise) — both new stages are
smoothing/regression over at most a few dozen points, negligible next to a
single inversion (~500-800 ms). Verified with no regressions (no exceptions,
no out-of-range or non-finite t0) across all 4 `data/*.txt` files and all 56
traces in `data_series/series_01`.

Also worth doing, not yet implemented: since `scale` in `despike_mask` is
global (§1.1), a second, **local** Hampel pass restricted to a window around
the coarse peak (its own, tighter `n_sigma`) would let despiking itself catch
peak-adjacent spikes without loosening detection elsewhere on the trace. Lower
priority now that coarse localization no longer depends on despiking alone.

### Option 2 — Robust loss inside the existing nested search (~free: no material cost increase)

Keep `T0_MODE="fit"`'s architecture exactly as-is (bounded Brent search,
seed ± bound, full inversion per candidate), but change the **objective**
from plain RMS to a robust residual summary — Huber loss, or a trimmed RMS
that drops the worst ~1-2% of pointwise residuals before averaging.
Implementation is a few lines in `_crli_scan_at_t0` / `_invert_at_t0` (and
`_resolve_trace_t0`'s 2-point misfit). The number and cost of inner
evaluations is unchanged (~14 × ~500-800 ms), so this is effectively free
relative to what's already being paid — it just stops a spike that lands
*inside* the fit window (as opposed to right on the peak sample) from biasing
which t0 the search prefers.

Doesn't fix a bad seed by itself — the search still starts, and stays
bounded, around whatever §1.2 produced. Pairs naturally with Option 1, which
does fix the seed.

### Option 3 — Seed-independent search, optionally with a Bayesian uncertainty layer (expensive → very expensive)

Stop trusting the seed's neighbourhood at all. Two escalating variants:

**3a. Coarse-to-fine global search (expensive).** Replace the tight `seed ±
5 s` Brent bracket with a coarse grid across a much wider window (e.g.
±20 s), each point scored by the full inversion, then Brent-polish the best
region. A ~40-point coarse grid + ~15-point polish ≈ 55 evals — about **4×
today's "fit" cost**: roughly **25-40 s/trace** for CRLI/Laplace at current
grid/α settings (measured ~474-769 ms/eval), versus ~6.8-8.8 s today. Across
the 56-trace `data_series/series_01` batch, that's the difference between a
~6-8 minute run and a ~25-35 minute run if applied to every trace in
single-trace mode. Pair with Option 2's robust objective so a spike anywhere
in the now much wider search range can't create a false global optimum.

**3b. Joint Bayesian estimation with uncertainty (very expensive).** Treat t0
as a parameter with a posterior, not a point estimate: MCMC (e.g. `emcee`)
sampling t0 — jointly with the free-fraction/amplitude nuisance parameters —
using a robust likelihood (Student-t rather than Gaussian, so residual
outliers are downweighted automatically rather than needing a separate
despike pass). Each MCMC step needs one full inversion, so cost scales as
`walkers × steps × ~500-800 ms`; e.g. 32 walkers × 500 steps ≈ 16,000 evals
≈ **2+ hours per trace**. Not something to run on every trace in a batch, but
it produces an actual credible interval on t0 that could be propagated into
the reported Rh-distribution uncertainty — something nothing in the current
pipeline does, since t0 is always treated as exactly known once resolved.
Best used as an occasional diagnostic on a hand-picked problem trace (like
`0p05perc`) rather than a batch default.

Either variant is architecture-dependent in cost: applied to
`global_series`'s per-trace t0 resolution (§1.3, ~0.5 ms/eval today) it would
be **~1000× cheaper** than applied to CRLI/Laplace's full-grid inversion
(~474-769 ms/eval) — worth remembering if this is wanted for the series
workflow specifically rather than the single-trace scripts.

## 3. Suggested path

Option 1 (§2, implemented) is the one that actually addresses what happened
on `0p05perc`: the *seed*, not the nested search, is what a bad sample
corrupts, and fixing it costs single-digit milliseconds — noise next to the
~0.5-0.8 s a single inversion already costs. Option 2 is a near-free add-on
worth doing next time this code is touched. Option 3 is worth keeping in
reserve for auditing a specific problem trace, not as a default — its cost
multiplies the part of the pipeline that's already the most expensive (14× a
single fit today; 3a pushes that to ~55×, 3b to ~16,000×).
