import numpy as np

def population_risk(thresholds, noise=0.1):
    thresholds = np.asarray(thresholds, dtype=float)
    if np.any((thresholds < 0) | (thresholds > 1)) or not 0 <= noise <= 0.5:
        raise ValueError('Thresholds must be in [0, 1], noise in [0, 0.5].')
    return noise + (1 - 2 * noise) * np.abs(thresholds - 0.5)

def empirical_risks(x, y, thresholds):
    x, y, thresholds = np.asarray(x), np.asarray(y), np.asarray(thresholds)
    if x.ndim != 1 or y.shape != x.shape or len(x) == 0:
        raise ValueError('A nonempty one-dimensional sample is required.')
    return np.mean((x[:, None] >= thresholds[None, :]) != y[:, None], axis=0)

def uniform_bound(n, hypotheses, delta=0.05):
    if n <= 0 or hypotheses <= 0 or not 0 < delta < 1:
        raise ValueError('Positive n and class size, and delta in (0, 1), required.')
    return np.sqrt(np.log(2 * hypotheses / delta) / (2 * n))
