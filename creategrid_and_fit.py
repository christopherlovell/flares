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
    w[np.where(uplims==1)] = 1e-5

    chi = chisquare(phiobs, yy, phiobserr)

    #Calculating the reduced chisquare and downwaiting any upper/lower limits
    chi = np.sum(chi*w)/(len(chi)-1)

    return chi

def read_data_and_fit(z, BC_fac):

    if z in [5, 6, 7]:
        data = np.genfromtxt(f"./Obs_data/uv_lum_Bouw15_z{z}.txt", delimiter=',', skip_header=1)
        M, phi, phi_err, uplims = data[:,0], data[:,1], data[:,2], data[:,3]
    elif z == 8:
        data = np.genfromtxt(f"./Obs_data/uv_lum_Bouw15_z{z}.txt", delimiter=',', skip_header=1)
        M, phi, phi_err, uplims = data[:,0], data[:,1], data[:,2], data[:,3]
        data = np.genfromtxt(f"./Obs_data/uv_lum_Bowler19_z{z}.txt", delimiter=',', skip_header=1)
        M1, M_err, phi1, phi_err1, uplims1 = data[:,0], data[:,1], data[:,2]*1e-6, data[:,3]*1e-6, data[:,4]
        M = np.append(M, M1)
        phi = np.append(phi, phi1)
        phi_err = np.append(phi_err, phi_err1)
        uplims = np.append(uplims, uplims1)
    elif z == 9:
        data = np.genfromtxt(f"./Obs_data/uv_lum_Bouw16_z{z}.txt", delimiter=',', skip_header=1)
        M, phi, phi_up, phi_low, uplims = data[:,0], data[:,1]*1e-3, data[:,2]*1e-3, data[:,3]*1e-3, data[:,4]
        phi_err = (phi_up + phi_low)/2.
        data = np.genfromtxt(f"./Obs_data/uv_lum_Bowler19_z{z}.txt", delimiter=',', skip_header=1)
        M1, M_err, phi1, phi_err1, uplims1 = data[:,0], data[:,1], data[:,2]*1e-6, data[:,3]*1e-6, data[:,4]
        M = np.append(M, M1)
        phi = np.append(phi, phi1)
        phi_err = np.append(phi_err, phi_err1)
        uplims = np.append(uplims, uplims1)

    ok = np.where((M < -18) & (uplims==0))[0]
    M = M[ok]
    phi = phi[ok]
    phi_err = phi_err[ok]
    uplims = uplims[ok]

    try:
        data = np.load(F'FUVLF_{extinction}_z{z}_BC{BC_fac}.npz')

    except Exception as e:
        print ("Create grid first!")
        print (e)

    bincen = data['Mag']
    hists = data['LF']
    Kappas = data['Kappas']
    ok = np.where(bincen < -18)[0]
    bincen = bincen[ok]
    hists = hists[:,ok]
    binwidth = bincen[1] - bincen[0]
    vol = (4/3)*np.pi*(14/h)**3

    phi_resim = hists


    rchi = np.array([])

    for ii in range(len(Kappas)):

        rchi = np.append(rchi, get_chisquare(M, phi, phi_err, uplims, bincen, phi_resim[ii]))

    return rchi

run_type, BC_fac, ext_curve = sys.argv[1], np.float(sys.argv[2]), sys.argv[3]

h = 0.6777
sims = np.arange(0,40)

"""

    Fitting is done at z = 5 against Bouwens et al. 2015, hence z is chosen as 5 and tag = '010_z005p000'

"""

z = 5
tag = '010_z005p000'
vol = (4/3)*np.pi*(14/h)**3
log10t_BC = 7.
extinction = ext_curve

bins = np.arange(-24, -16, 0.2)
bincen = (bins[1:]+bins[:-1])/2.
binwidth=bins[1:]-bins[:-1]

# Range of Kappa explored
Kappas = np.arange(0.01, 0.016, 0.00005)#np.arange(0.005, 0.03, 0.00025)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0: print (F"MPI size = {size}")

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

    print (F'Rank {rank} will be running {len(thisok)} task(s)')

    for ii, jj in enumerate(thisok):

        data, num, derr = get_lum_all(Kappas[jj], tag, BC_fac, bins = bins, LF = True, filters = ['FAKE.TH.FUV'], log10t_BC = log10t_BC, extinction = extinction)
        hist[jj] = data/(binwidth*vol)
        err[jj] = derr/(binwidth*vol)

    comm.Reduce(hist, hists, op=MPI.SUM, root=0)
    comm.Reduce(err, errs, op=MPI.SUM, root=0)

    if rank == 0:
        np.savez('FUVLF_{}_z{}_BC{}.npz'.format(extinction, z, BC_fac), LF = hists, err = errs, Mag = bincen, Kappas = Kappas)

if rank == 0:
    rchi = read_data_and_fit(z, BC_fac)
    print ('Kappa = ', Kappas[np.where(rchi==np.min(rchi))[0]])

    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize=(6, 6), sharex=True, sharey=True, facecolor='w', edgecolor='k')

    axs.plot(Kappas, rchi)
    axs.scatter(Kappas[np.where(rchi==np.min(rchi))[0]], np.min(rchi), marker = 'x', color = 'red')
    axs.text(Kappas[int(len(Kappas)/4)], np.median(rchi), rF"$\kappa$ = {Kappas[np.where(rchi==np.min(rchi))[0]]}", fontsize = 15)
    axs.set_xlabel(r"$\kappa$'s", fontsize = 15)
    axs.set_ylabel(r"reduced $\chi^2$")
    axs.grid(True)
    fig.tight_layout()
    plt.savefig(F'kappa_vs_rchi_{extinction}_BC{BC_fac}_z{z}.png')
    plt.close()
