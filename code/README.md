# Core-Flood Modeling Source

This implementation provides a runnable Bayesian inverse-modeling workflow. The demonstration constructs observations from synthetic functions and random innovations. Its scores and charts describe the synthetic run. Experimental aggregate results are presented in the [project overview](../README.md).

## Run

Use Python 3.12 or newer. From this directory:

```bash
python -m pip install -r requirements.txt
python run_package.py --output results
```

For a shorter execution check:

```bash
python run_package.py --output smoke-results --workers 2 --burn 30 --draws 55 --synthetic-replicates 2 --synthetic-burn 30 --synthetic-draws 55
```

The short run checks execution. Use the default draws and inspect the convergence diagnostics for quantitative inference. Each run saves a consolidated workbook, posterior chains, a run manifest, and seven figures in PNG, SVG, and PDF formats.

## Implementation

| Module | Purpose |
| --- | --- |
| `inverse.py`, `diagnostics.py` | Constrained Gibbs sampling, nested pressure models, prediction, split R-hat and effective sample size |
| `demo_config.py`, `synthetic_inputs.py` | Synthetic experiment configuration and observation generator |
| `run_advanced.py` | Blocked validation, baselines, CRPS, interval scores and prior sensitivity |
| `design_audit.py`, `synthetic_repeats.py` | Identifiability, sampling design and repeated synthetic checks |
| `plot_advanced.py`, `plot_comparisons.py` | Forecast, error, uncertainty, and comparison figures |
| `run_package.py`, `verify_results.py`, `export_results.py`, `result_tables.py` | Orchestration, validation and workbook export |
