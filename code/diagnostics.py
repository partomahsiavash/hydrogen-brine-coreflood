import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

def _split(values):
    h = values.shape[1] // 2
    return np.concatenate((values[:, :h], values[:, -h:]))


def _rank(values):
    return ndtri((rankdata(values).reshape(values.shape) - 0.375) / (values.size + 0.25))


def _variance(values):
    n = values.shape[1]
    within = np.mean(np.var(values, axis=1, ddof=1))
    pooled = within * (n - 1) / n + np.var(values.mean(1), ddof=1)
    return within, pooled


def _ess(values):
    m, n = values.shape
    within, pooled = _variance(values)
    if pooled <= 0:
        return float(m * n)
    centered = values - values.mean(1, keepdims=True)
    fft = np.fft.rfft(centered, n=2 * n, axis=1)
    autocov = np.fft.irfft(abs(fft)**2, axis=1)[:, :n].mean(0) / n
    rho = 1 - (within - autocov) / pooled
    rho[0] = 1
    pairs = rho[: n - n % 2].reshape(-1, 2).sum(1)
    stop = np.flatnonzero(pairs <= 0)
    pairs = np.minimum.accumulate(pairs[:stop[0]] if len(stop) else pairs)
    tau = max(1 / np.log10(m * n), -1 + 2 * pairs.sum())
    return float(m * n / tau)


def chain_diagnostics(values):
    split = _split(values)
    ranked = _rank(split)
    folded = _rank(np.abs(split - np.median(split)))
    rhat = max(np.sqrt(pooled / within) if within else np.nan
               for within, pooled in map(_variance, (ranked, folded)))
    lo, hi = np.quantile(split, [0.05, 0.95])
    tail = min(_ess((split <= lo).astype(float)), _ess((split <= hi).astype(float)))
    return rhat, _ess(ranked), tail
