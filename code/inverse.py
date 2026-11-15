from dataclasses import dataclass

import numpy as np
from scipy.special import log_ndtr, ndtri_exp

from diagnostics import chain_diagnostics
from demo_config import EDGES, SLOPE_SD, FOCAL


WIDTH = np.diff(EDGES)
PRESSURE_SD = np.array([1.0, .85, .65, .45, .16])


def recovery_design(x):
    return np.clip(np.asarray(x)[:, None] - EDGES[:-1], 0, WIDTH)


def pressure_design(x):
    x = np.asarray(x)
    return np.column_stack((np.ones(x.size), -np.expm1(-x/.012), -np.expm1(-x/.075),
                            -np.expm1(-x/.30), x))


def whiten(x, values, length):
    values = np.asarray(values, dtype=float)
    if not length:
        return values.copy()
    rho = np.exp(-np.diff(x) / length)
    vector_shape = (rho.size,) + (1,) * (values.ndim - 1)
    inverse_sd = (1 / np.sqrt(-np.expm1(-2 * np.diff(x) / length))).reshape(vector_shape)
    innovations = (values[1:] - rho.reshape(vector_shape) * values[:-1]) * inverse_sd
    return np.concatenate((values[:1], innovations), axis=0)


def truncated_normal(center, sd, low, high, rng):
    lo, hi = (low-center)/sd, (high-center)/sd
    if not lo < hi:
        return low
    if lo > 0:
        flipped = True
        lo, hi = -hi, -lo
    else:
        flipped = False
    u = rng.uniform(1e-14, 1-1e-14)
    log_p = np.logaddexp(np.log1p(-u) + log_ndtr(lo), np.log(u) + log_ndtr(hi))
    z = -ndtri_exp(log_p) if flipped else ndtri_exp(log_p)
    return float(np.clip(center+sd*z, low, high))


@dataclass(frozen=True)
class Series:
    x: np.ndarray
    y: np.ndarray
    length: float
    scale: float
    offset: float = 0.0


@dataclass(frozen=True)
class Fit:
    rf: Series | None
    pin: Series | None
    strict: bool
    linked: bool
    tail_width: float = 1.
    robust: bool = False

    def sample(self, burn, draws, seed):

        n = len(WIDTH) if self.rf is not None else 0
        if self.linked and (not n or self.pin is None):
            raise ValueError("The linked design requires both observation channels.")
        rf = self.rf
        rp = self.pin
        if n:
            rf_d = whiten(rf.x, recovery_design(rf.x), rf.length)
            rf_y = whiten(rf.x, rf.y, rf.length)
            rate_sd = SLOPE_SD.copy()
            rate_sd[6:] *= self.tail_width
            prior_prec = 1 / rate_sd**2
        else:
            rf_y = np.empty(0)
        if rp is not None:
            pin_y = whiten(rp.x, rp.y, rp.length)
            pin_d = whiten(rp.x, pressure_design(rp.x), rp.length)
            pin_r = (whiten(rp.x, 4 * recovery_design(rp.x), rp.length)
                     if self.linked else None)
        else:
            pin_y = np.empty(0)
        chain_keys = [*[f"rate_{i+1}" for i in range(n)],
                      *([f"pressure_beta_{i+1}" for i in range(len(PRESSURE_SD))] if rp is not None else []),
                      *(["gamma"] if self.linked else []),
                      *(["rf_sigma"] if n else []), *(["pin_sigma"] if rp is not None else [])]
        samples = np.empty((4, draws, len(chain_keys)))
        for c, state in enumerate(np.random.SeedSequence(seed).spawn(4)):
            rng = np.random.default_rng(state)
            r = np.minimum(.65, np.abs(rng.normal(size=n)) * rate_sd) if n else np.empty(0)
            if n and WIDTH @ r > .8:
                r *= .8 / (WIDTH @ r)
            pin_coef = np.zeros(len(PRESSURE_SD))
            gamma = 0.
            rf_var, pin_var = .018**2, .085**2
            rf_w, pin_w = np.ones(rf_y.size), np.ones(pin_y.size)
            if n:
                train_prec = rf_d.T @ rf_d
                train_info = rf_d.T @ rf_y
            for iteration in range(burn+draws):
                if self.linked:
                    d = np.column_stack((pin_d, pin_r @ r))
                    beta_sd = np.r_[PRESSURE_SD, .65]
                elif rp is not None:
                    d, beta_sd = pin_d, PRESSURE_SD
                if rp is not None:
                    precision = np.diag(1/beta_sd**2) + d.T @ (pin_w[:, None]*d) / pin_var
                    information = d.T @ (pin_w*pin_y) / pin_var
                    chol = np.linalg.cholesky(precision)
                    coef = np.linalg.solve(chol.T, np.linalg.solve(chol, information) +
                                           rng.standard_normal(len(beta_sd)))
                    pin_coef, gamma = coef[:len(PRESSURE_SD)], (coef[-1] if self.linked else 0.)
                if n:
                    if self.robust:
                        train_prec = rf_d.T @ (rf_w[:, None]*rf_d)
                        train_info = rf_d.T @ (rf_w*rf_y)
                    precision = np.diag(prior_prec) + train_prec / rf_var
                    information = train_info / rf_var
                    if self.linked:
                        design = gamma * pin_r
                        precision += design.T @ (pin_w[:, None]*design) / pin_var
                        information += design.T @ (pin_w*(pin_y - pin_d @ pin_coef)) / pin_var
                    off_diag = precision.copy()
                    np.fill_diagonal(off_diag, 0)
                    diag = np.diag(precision)
                    cap = WIDTH @ r
                    for i in rng.permutation(n):
                        high = max(0., (1-cap)/WIDTH[i] + r[i])
                        if self.strict:
                            high = min(1., high)
                        draw = truncated_normal((information[i]-off_diag[i] @ r)/diag[i],
                                                1/np.sqrt(diag[i]), 0., high, rng)
                        cap += WIDTH[i]*(draw-r[i])
                        r[i] = draw
                    residual = rf_y - rf_d @ r
                    rf_var = (2*.018**2 + .5*rf_w @ residual**2) / rng.gamma(3 + .5*rf_y.size)
                    if self.robust:
                        rf_w = rng.gamma(2.5, 1/(2+.5*residual**2/rf_var))
                if rp is not None:
                    residual = pin_y - pin_d @ pin_coef
                    if self.linked:
                        residual -= gamma*(pin_r @ r)
                    pin_var = (2*.085**2 + .5*pin_w @ residual**2) / rng.gamma(3 + .5*pin_y.size)
                    if self.robust:
                        pin_w = rng.gamma(2.5, 1/(2+.5*residual**2/pin_var))
                if iteration >= burn:
                    samples[c, iteration-burn] = (
                        *r, *(pin_coef if rp is not None else []), *([gamma] if self.linked else []),
                        *([np.sqrt(rf_var)] if n else []), *([np.sqrt(pin_var)] if rp is not None else []))
        return samples, chain_keys

    def mean(self, draws, keys, x, channel):
        keys = {name: i for i, name in enumerate(keys)}
        if channel == "RF" or self.linked:
            r = draws[..., [keys[f"rate_{i+1}"] for i in range(len(WIDTH))]]
            recovery = r @ recovery_design(x).T
            if channel == "RF":
                return recovery
        pressure = draws[..., [keys[f"pressure_beta_{i+1}"] for i in range(len(PRESSURE_SD))]]
        trend = pressure @ pressure_design(x).T
        return trend + 4*draws[..., keys["gamma"], None]*recovery if self.linked else trend

    def predictive(self, sample, keys, x, channel, seed):

        rng = np.random.default_rng(seed)
        data = self.rf if channel == "RF" else self.pin
        flat = sample.reshape(-1, sample.shape[-1])
        draws = flat[rng.choice(len(flat), min(len(flat), 2500), replace=False)]
        mean = self.mean(draws, keys, x, channel)
        last = data.x[-1]
        if np.any(np.diff(x) <= 0) or np.any(x <= last):
            raise ValueError("Forecast coordinates must increase after the last measurement.")
        residual = data.y[-1] - self.mean(draws, keys, np.array([last]), channel)[:, 0]
        sigma = draws[:, keys.index("rf_sigma" if channel == "RF" else "pin_sigma")]
        forecast = np.empty_like(mean)
        for i, step in enumerate(np.diff(np.r_[last, x])):
            rho = np.exp(-step/data.length) if data.length else 0.
            innovation = (rng.standard_t(4, len(draws)) if self.robust
                          else rng.standard_normal(len(draws)))
            residual = rho*residual + np.sqrt(1-rho*rho)*sigma*innovation
            forecast[:, i] = mean[:, i] + residual
        return (mean*data.scale + data.offset, forecast*data.scale + data.offset)


def prior_draws(tail_width, strict, seed, n=16000):
    rng = np.random.default_rng(seed)
    sd = SLOPE_SD.copy()
    sd[6:] *= tail_width
    accepted = []
    count = 0
    for _ in range(40):
        values = np.abs(rng.normal(size=(2*n,len(sd))))*sd
        valid = values @ WIDTH <= 1
        if strict:
            valid &= values.max(axis=1) <= 1
        selected = values[valid]
        accepted.append(selected)
        count += selected.shape[0]
        if count >= n:
            return np.concatenate(accepted)[:n]
    raise RuntimeError("Unable to sample the constrained prior with the specified precision.")


def diagnostics(sample, keys):
    rows = []
    quantities = {name: sample[..., i] for i, name in enumerate(keys)}
    if "rate_1" in keys:
        r = sample[..., :len(WIDTH)]
        quantities.update({f"RF at {x:.2f} PV": r @ a for x, a in zip(FOCAL, recovery_design(FOCAL))})
        quantities["Maximum latent slope"] = r.max(axis=-1)
    for name, values in quantities.items():
        rh, bulk, tail = chain_diagnostics(values)
        lo, md, hi = np.quantile(values, [.05,.50,.95])
        rows.append((name, lo, md, hi, rh, bulk, tail, bool(rh < 1.01 and min(bulk,tail) >= 400)))
    return rows
