# Learning Theory Lab

Educational example created for an academic homepage prototype. This repository
uses synthetic data and makes no claim of a published research contribution.
The example is not evidence of any person's education, employment, or authorship.

## Finite-class uniform convergence

Inputs follow `X ~ Uniform(0, 1)`. Clean labels are `1[X >= 0.5]` and each label
is flipped independently with probability `eta = 0.1`. The hypothesis class is
41 thresholds `h_t(x) = 1[x >= t]` fixed before sampling.

The exact population risk is `R(t) = eta + (1 - 2 eta) |t - 0.5|` for `t` in
`[0, 1]`. For every fixed sample size, Hoeffding's inequality and a union bound
imply

```text
P(max_t |R_hat(t) - R(t)| > sqrt(log(2 H / delta) / (2 n))) <= delta.
```

The plot compares the empirical 95th percentile of the maximum gap across 128
independent trials with this analytical bound. Its shaded interval is the
10th–90th trial percentiles, not a confidence interval. The bound applies
separately at each sample size; no simultaneous claim over the size grid is made.

The bound is an upper guarantee, so conservativeness is expected. Finite Monte
Carlo observations neither prove nor disprove its stated probability.

## Links

- Repository: https://github.com/km1120-code/learning-theory-lab
- Interactive homepage: https://km1120-code.github.io/
- Project explanation: https://km1120-code.github.io/projects/learning-theory-lab.html

## Run

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo.py
python -m unittest discover -s tests -v
```

The demo writes a PNG figure, a CSV of the plotted observations, and a JSON
summary to `results/`. Random seeds and experimental settings are stored in the
summary. Results depend slightly on numerical-library versions.

## Files

- `core.py`: mathematical functions and algorithms.
- `demo.py`: reproducible experiment and plotting.
- `tests/test_core.py`: numerical and edge-case checks.
- `results/`: outputs from the provided experiment.

## Scope

These small CPU experiments illustrate specified models. Their observations do
not establish behavior on real data, general-purpose optimizers, or deployed
communication systems. Code is available under the MIT license.
