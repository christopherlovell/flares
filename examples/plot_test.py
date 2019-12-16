
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import flares.plt as fplt


# ------------------------------ redshift evolution plot

fig, ax = fplt.single()

x = np.array([7,12])

redshifts = np.array([5,6,7,8,9,10])

norm = mpl.colors.Normalize(vmin=redshifts.min(), vmax=redshifts.max()+1) # +1 gets rid of the yellow colour

for z, m in zip(redshifts, np.linspace(0.25,1.0,len(redshifts))):
    ax.plot(x, (x-7)*m+27.5+m, c=cm.plasma(norm(z)), lw=1, label=rf'$z={z}$')

ax.set_ylim([27, 32])
ax.set_xlim([7, 12])

ax.set_ylabel(r'$\log_{10}(L_{\nu}/{\rm erg\ s^{-1}\ Hz^{-1})}$')
ax.set_xlabel(r'$\log_{10}(M_{\star}/{\rm M_{\odot}})$')

ax.grid(True)
ax.legend()

fig.savefig('redshift.pdf')






# ------------------------------  density dependence plot

fig, ax = fplt.single()

# --- line, fill_between

x = np.array([7,12])
y = np.array([28.5,31.5])

ax.fill_between(x,y-0.2,y+0.2, color='k', alpha=0.2)
ax.plot(x,y, c='k', lw=2, label='weighted total')

# ---

densities = np.linspace(-0.5,0.5,5)
norm = mpl.colors.Normalize(vmin=densities.min(), vmax=densities.max())

for density in densities:
    ax.plot(x,y+density*2+0.1, c=cm.viridis(norm(density)), lw = 1, label = rf'$\log_{{10}}(1+\delta)\in [{density}]$')


ax.set_ylim([27, 32])
ax.set_xlim([7, 12])

ax.set_ylabel(r'$\log_{10}(L_{\nu}/{\rm erg\ s^{-1}\ Hz^{-1})}$')
ax.set_xlabel(r'$\log_{10}(M_{\star}/{\rm M_{\odot}})$')

ax.grid(True)

ax.legend()

fig.savefig('density.pdf')







#
# # --- scatter1
#
# N = 50
# x = 8 + np.random.randn(N)/5
# y = 30 + np.random.randn(N)/5
# z = x**2 + y**2
#
# ax.scatter(x,y,c=z,cmap=cm.plasma,s=10)
