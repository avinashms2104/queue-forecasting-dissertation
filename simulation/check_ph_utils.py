"""Crosscheck ph_utils.py against the Gamma sampler already used in mph1_simulator.py."""
import numpy as np
from ph_utils import erlang, exponential

rng = np.random.default_rng(123)
n = 1_000_000

# 1. Erlang-2 service (mean 1): PH sampler vs numpy gamma(shape=2, scale=0.5), as in mph1_simulator.py
ph = erlang(2, 1.0).sample(n, rng)
gm = rng.gamma(shape=2, scale=0.5, size=n)
print("Erlang-2, mean 1        exact    PH sampler   numpy gamma")
print(f"  mean                  {1.0:.4f}   {ph.mean():.4f}       {gm.mean():.4f}")
print(f"  SCV                   {0.5:.4f}   {ph.var()/ph.mean()**2:.4f}       {gm.var()/gm.mean()**2:.4f}")
for q in [0.1, 0.5, 0.9, 0.99]:
    print(f"  {int(q*100)}th percentile      {'':6}   {np.quantile(ph, q):.4f}       {np.quantile(gm, q):.4f}")

# 2. Erlang-2 arrivals at rho = 0.7 (mean 1/0.7)
a = erlang(2, 1 / 0.7)
x = a.sample(n, rng)
print(f"\nErlang-2 arrivals, rho=0.7: exact mean {a.mean:.4f}, sampled {x.mean():.4f}; "
      f"exact SCV {a.scv:.4f}, sampled {x.var()/x.mean()**2:.4f}")

# 3. Exponential special case (should behave exactly like M/M/1 arrivals)
e = exponential(1.0).sample(n, rng)
print(f"Exponential mean 1: sampled mean {e.mean():.4f}, SCV {e.var()/e.mean()**2:.4f} (both should be ~1)")