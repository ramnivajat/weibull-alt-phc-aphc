# Theoretical_Datasets.py
"""50 censoring settings shared by PHC and APHC.

Preserves the original 21 settings, including keys, removal arrays and t.
Adds 29 settings to complete a fixed-n comparison grid at five proportions.
m/n is the planned failure proportion, not the censored proportion.
"""
import numpy as np
from collections import Counter
from fractions import Fraction

# 1. Original 15 settings (unchanged).
n1 = 58
R1 = np.concatenate([np.ones(24, dtype=int), [9]])
t1 = 2.73
n11 = 75
R11 = np.full(25, 2, dtype=int)
t11 = 2.73
n12 = 58
R12 = np.concatenate([np.array([16]), np.zeros(7, dtype=int), np.ones(17, dtype=int)])
t12 = 2.73
n2 = 80
R2 = np.ones(40, dtype=int)
t2 = 2.73
n21 = 120
R21 = np.full(40, 2, dtype=int)
t21 = 2.73
n22 = 80
R22 = np.concatenate([np.array([5, 6]), np.zeros(9, dtype=int), np.ones(29, dtype=int)])
t22 = 2.73
n3 = 100
R3 = np.ones(50, dtype=int)
t3 = 2.73
n4 = 150
R4 = np.full(50, 2, dtype=int)
t4 = 2.73
n41 = 100
R41 = np.tile(np.array([0, 2]), 25)
t41 = 2.73
n5 = 150
R5 = np.ones(75, dtype=int)
t5 = 2.73
n51 = 225
R51 = np.full(75, 2, dtype=int)
t51 = 2.73
n52 = 150
R52 = np.concatenate([np.zeros(50, dtype=int), np.full(25, 3, dtype=int)])
t52 = 2.73
n6 = 200
R6 = np.ones(100, dtype=int)
t6 = 2.73
n61 = 300
R61 = np.full(100, 2, dtype=int)
t61 = 2.73
n62 = 200
R62 = np.concatenate([np.zeros(98, dtype=int), np.array([50, 50])])
t62 = 2.73

# Keep entry order n, R, t for compatibility with .values() unpacking.
datasets = {
    'm25': {'n': n1, 'R': R1, 't': t1},
    'm25n': {'n': n11, 'R': R11, 't': t11},
    'm25nm': {'n': n12, 'R': R12, 't': t12},
    'm40': {'n': n2, 'R': R2, 't': t2},
    'm40n': {'n': n21, 'R': R21, 't': t21},
    'm40nm': {'n': n22, 'R': R22, 't': t22},
    'm50': {'n': n3, 'R': R3, 't': t3},
    'm50n': {'n': n4, 'R': R4, 't': t4},
    'm50nm': {'n': n41, 'R': R41, 't': t41},
    'm75': {'n': n5, 'R': R5, 't': t5},
    'm75n': {'n': n51, 'R': R51, 't': t51},
    'm75nm': {'n': n52, 'R': R52, 't': t52},
    'm100': {'n': n6, 'R': R6, 't': t6},
    'm100n': {'n': n61, 'R': R61, 't': t61},
    'm100nm': {'n': n62, 'R': R62, 't': t62},
}

# 2. Previously added six settings (unchanged).
datasets['m25_p50'] = {'n': 50, 'R': np.ones(25, dtype=int), 't': 2.73}
for m in (25, 40, 50, 75, 100):
    datasets[f'm{m}_p25'] = {
        'n': 4 * m, 'R': np.full(m, 3, dtype=int), 't': 2.73,
    }

# 3. Revised 29 additions.
# Six fixed initial sample sizes x five exact proportions = 30 grid cells.
# (m=40, n=120) is already present as m40n, leaving 29 additions.
# The same deterministic, evenly spread removal rule is used throughout.
# R[i] = floor((i+1)*(n-m)/m) - floor(i*(n-m)/m).
# This gives constant removals when possible, otherwise adjacent integer
# counts spread over the failures. Above 50%, R contains zeros and ones.
def balanced_removals(m, n):
    """Return m nonnegative integer removals summing exactly to n-m."""
    if not isinstance(m, (int, np.integer)) or not isinstance(n, (int, np.integer)):
        raise ValueError('m and n must be integers.')
    if not 0 < m <= n:
        raise ValueError('Require 0 < m <= n.')
    cumulative = (np.arange(m + 1, dtype=np.int64) * (n - m)) // m
    return np.diff(cumulative)

GRID_N = (120, 180, 240, 360, 480, 600)
PROPORTIONS = ((1, 4), (1, 3), (1, 2), (2, 3), (3, 4))
existing_pairs = {(len(s['R']), s['n']) for s in datasets.values()}
added_count = 0
for n in GRID_N:
    for numerator, denominator in PROPORTIONS:
        if (n * numerator) % denominator:
            raise ValueError('Grid proportions must give integer m.')
        m = n * numerator // denominator
        if (m, n) in existing_pairs:
            continue
        name = f'm{m}_n{n}_balanced'
        if name in datasets:
            raise ValueError(f'Duplicate dataset key: {name}')
        datasets[name] = {'n': n, 'R': balanced_removals(m, n), 't': 2.73}
        existing_pairs.add((m, n))
        added_count += 1

if added_count != 29 or len(datasets) != 50:
    raise ValueError(f'Expected 29 additions and 50 total; got {added_count}, {len(datasets)}.')

# 4. Validate on import, before any simulation starts.
def validate_datasets(settings):
    for name, setting in settings.items():
        if list(setting) != ['n', 'R', 't']:
            raise ValueError(f'{name}: expected entry order n, R, t.')
        n, R, t = setting.values()
        R = np.asarray(R)
        if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)):
            raise ValueError(f'{name}: n must be an integer.')
        if R.ndim != 1 or not np.issubdtype(R.dtype, np.integer):
            raise ValueError(f'{name}: R must be a one-dimensional integer array.')
        m = len(R)
        if not 0 < m <= n or np.any(R < 0):
            raise ValueError(f'{name}: invalid m, n or negative removals.')
        if m + int(R.sum()) != n:
            raise ValueError(f'{name}: m + sum(R) must equal n.')
        if not np.isfinite(t) or t <= 0:
            raise ValueError(f'{name}: t must be positive and finite.')
        prior_removals = np.concatenate(([0], np.cumsum(R[:-1])))
        risk_counts = n - np.arange(m) - prior_removals
        if np.any(risk_counts < 1 + R):
            raise ValueError(f'{name}: insufficient units for a failure and removals.')

validate_datasets(datasets)

# 5. Display the design when run directly; imports remain silent.
if __name__ == '__main__':
    unique_pairs = {(len(s['R']), s['n']) for s in datasets.values()}
    counts = Counter(Fraction(len(s['R']), s['n']) for s in datasets.values())
    print(f'Total censoring settings: {len(datasets)}')
    print(f'Unique (m, n) pairs: {len(unique_pairs)}')
    print('Planned failure proportions:')
    for proportion, count in sorted(counts.items()):
        print(f'  {float(proportion):.2%}: {count} settings')
    print()
    for number, (name, setting) in enumerate(datasets.items(), start=1):
        n, R, t = setting.values()
        m = len(R)
        print(f'{number:02d}. {name}: n={n}, m={m}, planned m/n={m/n:.2%}, '
              f'sum(R)={int(R.sum())}, t={t}')
