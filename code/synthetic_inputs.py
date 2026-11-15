import numpy as np
import pandas as pd

from demo_config import CASES, EDGES


def simulate_innovations(x, scale, length, rng):
    values = rng.normal(0., scale, len(x))
    for i, step in enumerate(np.diff(x), 1):
        rho = np.exp(-step / length)
        values[i] = rho * values[i - 1] + np.sqrt(1 - rho**2) * values[i]
    return values


def synthetic_grid():
    return {"RF": np.linspace(.04, EDGES[-1], 47),
            "Pinlet": np.linspace(.01, EDGES[-1], 123)}


def make_observations(seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i, case in enumerate(CASES):
        for channel, x in synthetic_grid().items():
            r = (.33 + .09*i) * (-np.expm1(-np.maximum(x - .08*i, 0.) / (.50 + .38*i)))
            if channel == "RF":
                y = 100 * (r + simulate_innovations(x, .008, .04, rng))
            else:
                y = 60 * (.20 - .45*np.expm1(-x/.08) + (.17 + .16*i)*r +
                          .025*x + simulate_innovations(x, .025, .06, rng))
            rows.extend((case, channel, a, b, j)
                        for j, (a, b) in enumerate(zip(x, y), 1))
    return pd.DataFrame(rows, columns=["Condition", "Channel", "PV", "Observed", "Source row"])


def balance_audit(observations):
    rows = []
    for case, frame in observations.query("Channel == 'RF'").groupby("Condition", sort=False):
        x = frame.PV.to_numpy()
        delta = np.diff(frame.Observed.to_numpy()) / 100
        slope = delta / np.diff(x)
        rows.extend((case, a, b, change, rate, rate > 1 + 1e-12)
                    for a, b, change, rate in zip(x[:-1], x[1:], delta, slope))
    return pd.DataFrame(rows, columns=["Condition", "PV start", "PV end", "RF change fraction",
                                      "Interval RF slope", "Exceeds unit slope"])
