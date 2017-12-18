import matplotlib.pyplot as plt
from matplotlib import cm, colors
import scipy.special as sp
import numpy as np

def plot_spherical_harmonic(l, m, xmesh=100, ymesh=50):
    # Coordinate arrays for the graphical representation
    x = np.linspace(-np.pi, np.pi, 100)
    y = np.linspace(-np.pi/2, np.pi/2, 50)
    X, Y = np.meshgrid(x, y)

    # Spherical coordinate arrays derived from x, y
    # Necessary conversions to get Mollweide right
    phi = x.copy()    # physical copy
    phi[x < 0] = 2 * np.pi + x[x<0]
    theta = np.pi/2 - y
    PHI, THETA = np.meshgrid(phi, theta)

    SH_SP = sp.sph_harm(m, l, PHI, THETA).real    # Plot just the real part
    xlabels = ['$210^\circ$', '$240^\circ$','$270^\circ$','$300^\circ$','$330^\circ$',
               '$0^\circ$', '$30^\circ$', '$60^\circ$', '$90^\circ$','$120^\circ$', '$150^\circ$']

    ylabels = ['$165^\circ$', '$150^\circ$', '$135^\circ$', '$120^\circ$', 
               '$105^\circ$', '$90^\circ$', '$75^\circ$', '$60^\circ$',
               '$45^\circ$','$30^\circ$','$15^\circ$']
               

    fig, ax = plt.subplots(subplot_kw=dict(projection='mollweide'), figsize=(10,8))
    im = ax.pcolormesh(X, Y , SH_SP, cmap=cm.jet)
    ax.set_xticklabels(xlabels, fontsize=14)
    ax.set_yticklabels(ylabels, fontsize=14)
    ax.set_title('$Y^{'+str(l)+"}_{"+str(m)+'}$', fontsize=20)
    ax.set_xlabel(r'$\phi$', fontsize=20)
    ax.set_ylabel(r'$\theta$', fontsize=20)
    ax.grid()

    fig.colorbar(im, orientation='horizontal', cmap=cm.jet)