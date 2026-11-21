from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import *

OUT = Path(__file__).parent / 'results'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': '#34433e', 'text.color': '#34433e',
                     'axes.edgecolor': '#b6beb8', 'xtick.color': '#64726b',
                     'ytick.color': '#64726b', 'figure.facecolor': '#faf9f5',
                     'axes.facecolor': '#faf9f5', 'savefig.facecolor': '#faf9f5'})
GREEN, SAND, BLUE = '#235b48', '#b77c45', '#6b889a'

rng = np.random.default_rng(2026)
thresholds = np.linspace(0, 1, 41)
sizes = np.array([32, 64, 128, 256, 512, 1024, 2048])
trials, noise, delta = 128, 0.1, 0.05
risk = population_risk(thresholds, noise)
rows = []
for n in sizes:
    gaps = []
    for _ in range(trials):
        x = rng.uniform(size=n)
        y = (x >= 0.5) ^ (rng.random(n) < noise)
        gaps.append(float(np.max(np.abs(empirical_risks(x, y, thresholds) - risk))))
    q10, median, q90, q95 = np.quantile(gaps, [0.1, 0.5, 0.9, 0.95])
    bound = float(uniform_bound(n, len(thresholds), delta))
    rows.append({'n': int(n), 'q10': q10, 'median': median, 'q90': q90, 'q95': q95,
                 'bound': bound, 'violation_rate': float(np.mean(np.array(gaps) > bound))})
with (OUT / 'uniform-convergence.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
fig, ax = plt.subplots(figsize=(7.4, 4.4), layout='constrained')
ax.fill_between(sizes, [r['q10'] for r in rows], [r['q90'] for r in rows], color=GREEN, alpha=.10, label='10–90% of trials')
ax.plot(sizes, [r['q95'] for r in rows], 'o-', color=GREEN, label='95th percentile of max. gap', markersize=4)
ax.plot(sizes, [r['bound'] for r in rows], '--', color=SAND, label='Finite-class upper bound')
ax.set(xscale='log', yscale='log', xlabel='Training examples, n', ylabel='Maximum absolute risk gap', title='How does estimation error shrink?')
ax.grid(alpha=.15); ax.legend(frameon=False, fontsize=8)
fig.savefig(OUT / 'uniform-convergence.png', dpi=180)
summary = {'experiment': 'finite-class uniform convergence', 'seed': 2026, 'trials': trials,
           'noise': noise, 'delta': delta, 'hypotheses': len(thresholds), 'sizes': sizes.tolist(),
           'largest_n_q95': rows[-1]['q95'], 'largest_n_bound': rows[-1]['bound'],
           'interpretation': 'Synthetic Monte Carlo illustration; percentile band is not a confidence interval.'}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + chr(10))
print(json.dumps(summary))
