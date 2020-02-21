"""

    Creates Far-UVLF for a range of values of Kappa (see Vijayan et al. 2020 for more details)
    Uses MPI4PY to do the grid creation in parallel
    If input is 'Gridgen', first creates the grid and then fits,
    else looks for the grid output file and fits
    After creation fitting is done against Bouwens et al. 2015 at z = 5

"""

import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from mpi4py import MPI
from phot_modules import get_lum_all
from scipy import interpolate
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def chisquare(obs, exp, err):

    return ((obs - exp)**2)/(err**2)

def get_chisquare(Mobs, phiobs, phiobserr, uplims, xdata, ydata):

    ok = np.where(ydata > 0)

    f = interpolate.interp1d(xdata[ok], ydata[ok], fill_value="extrapolate")

    yy = f(Mobs)

    #Weights for observational datapoints that are upper/lower limits
    w = np.ones(len(uplims))
    w[np.where(uplims==1)] = 1e-3

    chi = chisquare(phiobs, yy, phiobserr)

    #Calculating the reduced chisquare and downwaiting any upper/lower limits
    chi = np.sum(chi*w)/(len(chi)-1)

    return chi

def read_data_and_fit(z):

    data = np.genfromtxt(f"./Obs_data/uv_lum_Bouw15_z{z}.txt", delimiter=',', skip_header=1)
    M, phi, phi_err, uplims = data[:,0], data[:,1], data[:,2], data[:,3]

    ok = np.where(M < -17.5)[0]
    M = M[ok]
    phi = phi[ok]
    phi_err = phi_err[ok]
    uplims = uplims[ok]

    try:
        data = np.load(F'FUVLF_z{z}.npz')

    except Exception as e:
        print ("Create grid first!")
        print (e)

    bincen = data['Mag']
    hists = data['LF']
    Kappas = data['Kappas']
    ok = np.where(bincen < -17.5)[0]
    bincen = bincen[ok]
    hists = hists[:,ok]
    binwidth = bincen[1] - bincen[0]
    vol = (4/3)*np.pi*(14/h)**3

    phi_resim = hists


    rchi = np.array([])

    for ii in range(len(Kappas)):

        rchi = np.append(rchi, get_chisquare(M, phi, phi_err, uplims, bincen, phi_resim[ii]))

    return rchi

run_type = sys.argv[1]

h = 0.6777
sims = np.arange(0,40)

"""

    Fitting is done at z = 5 against Bouwens et al. 2015, hence z is chosen as 5 and tag = '010_z005p000'

"""

z = 5
tag = '010_z005p000'
vol = (4/3)*np.pi*(14/h)**3

bins = np.arange(-24, -16, 0.2)
bincen = (bins[1:]+bins[:-1])/2.
binwidth=bins[1:]-bins[:-1]

# Range of Kappa explored
Kappas = np.arange(0.0, 0.2, 0.0025)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if run_type == 'Gridgen':

    hist = np.zeros((len(Kappas),len(bincen)))
    err = np.zeros((len(Kappas),len(bincen)))

    if rank == 0:
        hists = np.zeros((len(Kappas),len(bincen)))
        errs = np.zeros((len(Kappas),len(bincen)))

    else:
        hists = np.empty((len(Kappas),len(bincen)))
        errs = np.empty((len(Kappas),len(bincen)))

    part = int(len(Kappas)/size)

    if rank!=size-1:
        thisok = np.arange(rank*part, (rank+1)*part, 1).astype(int)

    else:
        thisok = np.arange(rank*part, len(Kappas), 1).astype(int)

    for ii, jj in enumerate(thisok):

        data, num, derr = get_lum_all(Kappas[jj], tag, bins = bins, LF = True, filters = ['FAKE.TH.FUV'])
        hist[jj] = data/(binwidth*vol)
        err[jj] = derr/(binwidth*vol)

    comm.Reduce(hist, hists, op=MPI.SUM, root=0)
    comm.Reduce(err, errs, op=MPI.SUM, root=0)

    if rank == 0:
        np.savez('FUVLF_z{}.npz'.format(z), LF = hists, err = errs, Mag = bincen, Kappas = Kappas)

if rank == 0:
    rchi = read_data_and_fit(z)
    print ('Kappa = ', Kappas[np.where(rchi==np.min(rchi))[0]])

    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize=(6, 6), sharex=True, sharey=True, facecolor='w', edgecolor='k')

    axs.plot(Kappas, rchi)
    axs.scatter(Kappas[np.where(rchi==np.min(rchi))[0]], np.min(rchi), marker = 'x', color = 'red')
    axs.text(Kappas[3*int(len(Kappas)/4)], int(np.max(rchi/2)), rF"$\kappa$ = {Kappas[np.where(rchi==np.min(rchi))[0]]}", fontsize = 15)
    axs.set_xlabel(r"$\kappa$'s", fontsize = 15)
    axs.set_ylabel(r"reduced $\chi^2$")
    axs.grid(True)
    fig.tight_layout()
    plt.savefig('kappa_vs_rchi.png')
    plt.close()
