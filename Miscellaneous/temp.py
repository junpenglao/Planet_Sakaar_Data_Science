import pymc3 as pm
import numpy as np
import matplotlib.pylab as plt

data = np.ma.masked_values([42] * 100 + [-1] * 100 + [42] * 100, value=-1)

with pm.Model() as model:
    s = pm.GaussianRandomWalk('s', sd=1e-2, shape=len(data))
    n = pm.Normal('n', mu=pm.math.exp(s), observed=data)

    trace = pm.sample()

plt.plot(trace['s'].T, alpha=0.1)

with pm.Model() as model:
    s = pm.GaussianRandomWalk('s', sd=1e-2, shape=len(data))
    n = pm.Poisson('n', mu=pm.math.exp(s), observed=data)
    step = pm.Slice(vars=model.vars[1])
    trace = pm.sample(step=step)

pm.traceplot(trace);
plt.plot(trace['s'].T, alpha=0.1);
