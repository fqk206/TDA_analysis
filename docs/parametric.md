# Parametric TDA fit — minimal test implementation

A plan for a small, single-trace parametric fitter that complements the free-form
CRLI / Laplace inversion. Global (multi-trace) fitting and bootstrap uncertainty
are **deferred to later**, but the design below keeps them easy to add.

Proposed script: `fit_parametric.py`, reusing `fit_crli_simple.prepare`, the Taylor
kernel, and helpers (one source of truth for data, physics, preprocessing).

## 1. Scope of the minimal version

- Fit **one trace** with a small, fixed number of **log-normal components** (start
  with two: a small "dye" component ~0.75 nm and a colloid component ~23 nm).
- Recover, per component: center `Rh`, log-width `σ`, and fraction.
- Produce the same distribution statistics as the inversion scripts so results are
  directly comparable.
- Out of scope for now: multi-trace global models, titration/binding links,
  bootstrap. The code is structured so these are wrappers added later (§7).

## 2. Forward model

Same physics as the inversion, but the distribution is parametric:

```
s(t) = Σ_k a_k ∫ g_k(Rh; μ_k, σ_k) K(t, Rh) dlnRh
```

- `g_k` = log-normal in Rh (center `μ_k`, log-width `σ_k`), normalised to unit area.
- `K(t, Rh)` = the existing Taylor kernel (`fit_crli_simple.rh_kernel`).
- Evaluate the integral on a fixed dense log-Rh grid with the existing trapezoid
  weights — identical machinery to the inversion, just a parametric `g_k` instead
  of a free vector.

## 3. Parametrization and VARPRO

- **Fit `ln μ` and `ln σ`**, not the raw values: positivity for free, scale-free,
  trivial bounds, well-conditioned.
- **Separate linear from nonlinear parameters (variable projection).** For fixed
  shapes (the `μ_k`, `σ_k`), the model is **linear in the amplitudes `a_k`**, so
  solve those with NNLS at every step. The nonlinear optimiser then only handles a
  handful of shape parameters (2 components → 4 shape params, + optional `t0`).
  This is the key to a stable fit and it reuses the NNLS solver already in the repo.
- Fractions = detector-weighted areas of the fitted amplitudes (same definition as
  the CRLI free fraction), so they line up with the other scripts.

## 4. Fit mechanics (single trace)

- `scipy.optimize.least_squares` (trust-region, bounded) over the shape parameters;
  residual = `(model − data)` on the normalised signal; amplitudes via NNLS inside.
- **Initial guesses** from the Laplace inversion (~0.75 nm, ~23 nm) and the slope
  estimate — good inits matter for nonlinear fits.
- **Optional nuisance parameter `t0`** (fit it, don't just take the preprocessing
  value): the structured apex residual we saw is probably a small `t0` / peak-shape
  error, and letting `t0` move lets it absorb that instead of biasing the sizes.
- Reuse the same fit-window controls as the inversion (`PEAK_FRACTION`, signal
  min/max, stride) so the comparison is apples-to-apples.

## 5. Config (minimal)

```
N_COMPONENTS        = 2
INIT_RH_NM          = [0.75, 23.0]     # initial centers
INIT_SIGMA_LN       = [0.05, 0.25]     # initial log-widths
RH_BOUNDS_NM        = (0.3, 200.0)     # per-center bounds
SIGMA_LN_BOUNDS     = (0.02, 1.5)
FIT_T0              = True              # refine t0 as a nuisance parameter
RH_GRID             = (0.2, 300.0, 400)  # dense grid for the model integral
OUTPUT_DIR          = "output/parametric"
```

Data pattern, time unit, physics, and `PEAK_FRACTION` come from `fit_crli_simple`.

## 6. Outputs

- **Summary CSV row** per trace: per-component `Rh`, `σ`, fraction; `t0`; residual
  RMS (raw and /peak, and /noise); convergence flag; plus the shared
  `distribution_stats` (median/mean/mode/percentiles) computed on the combined
  parametric distribution, for direct comparison with CRLI and Laplace.
- **Distribution CSV**: the fitted distribution sampled on the Rh grid
  (`Rh, pdf_per_lnRh, pdf_per_nm, cdf`), same format as the other scripts.
- **Run settings JSON**.

## 7. Designed for later (kept clean now, not built yet)

- A **pure forward-model function** `params -> signal(t)` and a **residual
  function returning a vector** — so a global fit just concatenates residuals over
  traces, and bootstrap just refits on resampled/synthetic data. No rework needed.
- **Parameter pack/unpack** through a small struct, so a later global layer can map
  a shared/trend/per-trace parameter vector onto each trace.
- Keep `t0`, peak scale, and amplitudes clearly tagged as **per-trace (local)** vs.
  the shapes as candidate **shared** parameters.

## 8. Diagnostic plots

The point of the parametric fit is a constrained, interpretable result, so the
plots should make it easy to judge whether the constraint is justified.

1. **Fit overlay** — data points + total model vs time (raw signal). The basic
   "does it fit" check.
2. **Residual vs time, with a ±noise band** — the most important diagnostic here.
   We *expect* structured residual at the apex; this plot shows whether the
   parametric model leaves more structure than the free-form inversion did (i.e.
   whether two log-normals are too rigid).
3. **Residual vs signal level** (binned, rising side) — exposes apex-vs-wing
   structure more clearly than the time view; this is the plot that first revealed
   the peak-shape mismatch.
4. **Signal-space decomposition** — total model plus each component's signal
   (dye, colloid) on the raw-signal axis, like the CRLI/Laplace decomposition
   plots. Shows how much each component contributes where.
5. **Distribution overlay vs the Laplace inversion** — the key cross-check: plot
   the two fitted log-normals (semilogx, PDF per lnRh) on top of the free-form
   Laplace distribution. If they agree on centers/areas, the breadth in the
   inversion was regularisation, not the sample; if they disagree, the parametric
   model may be too constrained (or wrong component count).
6. **(Optional) cost-vs-parameter slices** — 1-D profiles of the residual cost as
   each center/width is scanned around the optimum. A cheap, pre-bootstrap look at
   how well each parameter is determined and which ones are flat/degenerate (e.g.
   small-component width vs fraction). This previews what the later bootstrap will
   quantify.

## 9. Acceptance test

The minimal version "works" if, on `data/25ugml_ivALPS_DLS.txt`, it:

- converges to a small component near **~0.7–0.8 nm** and a colloid near
  **~23 nm**, with a small-component fraction of a few percent;
- reaches a residual comparable to the Laplace inversion (~1.5–2× noise), i.e. it
  does not leave large extra structure beyond the known apex mismatch;
- agrees with the Laplace distribution on the overlay plot (diagnostic 5).

If it meets these, the forward-model/residual interfaces from §7 are the only
things needed before adding the global and bootstrap layers.
