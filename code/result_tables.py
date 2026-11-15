from pathlib import Path
import pandas as pd

TABLES = {
    "metrics": "Metrics",
    "forecasts": "Forecasts",
    "diagnostics": "MCMC diagnostics",
    "prior_and_posterior": "Prior and posterior",
    "pressure_sensitivity": "Pressure sensitivity",
    "design_blind_intervals": "Blind intervals",
    "design_counterexample": "Counterexample",
    "design_sampling_gain": "Sampling design",
    "synthetic_repeated_summary": "Synthetic summary",
    "synthetic_repeated_trials": "Synthetic trials",
    "synthetic_truth": "Synthetic truth",
    "source_observations": "Source observations",
    "volume_audit": "Volume audit",
    "training_scales": "Training scales",
    "rate_grid": "Rate grid",
}

FILENAME = "H2_CoreFlood_Results.xlsx"


def read_result(directory, key):
    directory = Path(directory)
    path = directory / f"{key}.csv"
    if path.is_file():
        return pd.read_csv(path)
    return pd.read_excel(directory / FILENAME, sheet_name=TABLES[key])
