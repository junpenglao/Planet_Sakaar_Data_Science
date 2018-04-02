"""
https://twitter.com/junpenglao/status/928206574845399040
"""
import pymc3 as pm
import numpy as np
import matplotlib.pylab as plt

L = np.array([[2, 1]]).T
Sigma = L.dot(L.T) + np.diag([1e-2, 1e-2])
L_chol = np.linalg.cholesky(Sigma)

with pm.Model() as model:
    y = pm.MvNormal('y', mu=np.zeros(2), chol=L_chol, shape=2)
    tr0 = pm.sample(500, chains=1)
    tr1 = pm.fit(method='advi').sample(500)
    tr2 = pm.fit(method='fullrank_advi').sample(500)
    tr3 = pm.fit(method='svgd').sample(500)


plt.figure()
plt.plot(tr0['y'][:,0], tr0['y'][:,1], 'o', alpha=.1, label='NUTS')
plt.plot(tr1['y'][:,0], tr1['y'][:,1], 'o', alpha=.1, label='ADVI')
plt.plot(tr2['y'][:,0], tr2['y'][:,1], 'o', alpha=.1, label='FullRank')
plt.plot(tr3['y'][:,0], tr3['y'][:,1], 'o', alpha=.1, label='SVGD')
plt.legend();


"""
https://twitter.com/junpenglao/status/930826259734638598
"""
import matplotlib.pylab as plt
from mpl_toolkits import mplot3d
import numpy as np
import pymc3 as pm

def cust_logp(z):
    return -(1.-z[0])**2 - 100.*(z[1] - z[0]**2)**2

grid = np.mgrid[-2:2:100j,-1:3:100j]
Z = -np.asarray([cust_logp(g) for g in grid.reshape(2, -1).T])
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(grid[0], grid[1], Z.reshape(100,100), cmap='viridis',
                       linewidth=0, antialiased=False)

with pm.Model():
    pm.DensityDist('pot1', logp=cust_logp, shape=(2,))
    tr1 = pm.sample(500, step=pm.NUTS())['pot1']
    tr2 = pm.sample(500, step=pm.Metropolis())['pot1']
    tr3 = pm.fit (n=50000, method='fullrank_advi').sample(500)['pot1'] #VI, cause whynot

import matplotlib.pylab as plt
_, ax = plt.subplots(1,3,figsize=(15,5), sharex=True, sharey=True)
ax[0].imshow(Z.reshape(100,100), extent=[-1,3,-2,2,]);
ax[0].plot(tr1[:,1], tr1[:,0], 'ro-',alpha=.1)
ax[1].imshow(Z.reshape(100,100), extent=[-1,3,-2,2,]);
ax[1].plot(tr2[:,1], tr2[:,0], 'ro-',alpha=.1)
ax[2].imshow(Z.reshape(100,100), extent=[-1,3,-2,2,]);
ax[2].plot(tr3[:,1], tr3[:,0], 'ro', alpha=.1)
plt.tight_layout()

with pm.Model():
    pm.DensityDist('pot1', logp=cust_logp, shape=(2,))
    minimal=pm.find_MAP()
