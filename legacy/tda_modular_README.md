# Modular TDA fitting scripts

> **Archived.** This README documents the older multi-model modular pipeline,
> which now lives under `legacy/`. The current workflow is the single-file
> `fit_crli_simple.py`, with despiking and time binning factored into the shared
> `preprocess.py`. The file paths below (`tda_processing.py`, etc.) are relative
> to `legacy/`. See `docs/simpl.md` for the simplification that superseded this.

This package splits the Taylor Dispersion Analysis workflow into one shared processing module and one fitting script per model.

## Files

```text
tda_processing.py                  Shared file reading, preprocessing, spike handling, kernels, plotting, and batch output helpers.
fit_dual_gaussian.py               Fixed free fluorophore + one monodisperse colloid component.
fit_crli.py                        Fixed free fluorophore + smooth non-negative CRLI colloid distribution.
fit_parametric_distribution.py      Fixed free fluorophore + normal and/or lognormal colloid distribution.
```

The shared module is deliberately not a fitting script. It reads the taylorgram, estimates the baseline and peak time, detects narrow spikes, prepares the fitting masks, defines the Taylor kernels, and saves common outputs.

## Run by editing the script

Each fitting script has a small `USER SETTINGS` block near the top. For example, in `fit_dual_gaussian.py`:

```python
DATA_PATTERN = "*.txt"
OUTPUT_DIR = "tda_results"
INPUT_FORMAT = "fidaneo"
PEAK_FRACTION = 0.50
SAVE_PLOTS = True
```

Edit those values, put the scripts in the folder containing the data files, and press Run in your Python editor.

## Run from the command line

Dual-Gaussian fit, using the original rising-side-only setting:

```bash
python fit_dual_gaussian.py --data "*.txt" --output tda_results
```

Dual-Gaussian fit using more of the peak:

```bash
python fit_dual_gaussian.py --data "*.txt" --output tda_results --peak-fraction 0.75
```

CRLI fit:

```bash
python fit_crli.py --data "*.txt" --output tda_results
```

Parametric lognormal fit only:

```bash
python fit_parametric_distribution.py --data "*.txt" --output tda_results --distributions lognormal
```

Parametric normal and lognormal fits:

```bash
python fit_parametric_distribution.py --data "*.txt" --output tda_results --distributions normal lognormal
```

For quick testing without plots:

```bash
python fit_dual_gaussian.py --data "*.txt" --output tda_results --no-plots
```

## Output folders

The scripts save into separate model folders:

```text
tda_results/
  dual_gaussian/
  crli/
  parametric_normal/
  parametric_lognormal/
```

Each folder contains a model-specific summary CSV, a run-settings JSON file, distribution CSVs, preprocessing plots, raw fit plots, residual plots, decomposition plots, and distribution plots. The CRLI folder also contains alpha-scan diagnostics.

The old all-in-one comparison plots are not included here, because the point of the split is that each model can be run and shared independently. The summary and distribution CSV files can still be compared afterwards.

## Spike handling

Spike handling is shared by all models in `tda_processing.py`. By default, a narrow spike is excluded from preprocessing and fitting. A file is skipped if more than five spike events are detected.

Relevant settings are in the `Settings` dataclass:

```python
despike = True
max_spikes_allowed = 5
spike_threshold_mad = 10.0
spike_median_window_points = 11
spike_padding_points = 2
spike_min_prominence_fraction_of_signal_range = 0.15
```

Command-line overrides are also available:

```bash
--no-despike
--max-spikes 5
--spike-threshold 10
--spike-window 11
--spike-padding 2
--spike-min-prominence-fraction 0.15
```

## Input formats

The default is `fidaneo`, which reads FidaNeo `.txt` exports containing `<header>` and `<data>` blocks. It extracts pressure, method time, and capillary temperature from the `<meth>` table when available.

For simple files with only time and signal columns:

```bash
python fit_dual_gaussian.py --input-format two_column --data "*.csv" --two-column-delimiter ","
```

If a different instrument format is needed, add a reader function to `tda_processing.py` that returns a `Trace` object, then add one branch to `read_trace()`.

## Requirements

Python 3.10 or newer, plus:

```text
numpy
scipy
matplotlib
```
