import numpy as np
import pandas as pd
from mpi4py import MPI
import h5py
import re
import sys
import flares
from flares import get_recent_SFR
from phot_modules import get_lum_all
from phot_modules import get_flux_all
import FLARE.filters


def get_simtype(inp):

    if inp == 'FLARES':
        sim_type = 'FLARES'

    elif inp == 'REF' or inp == 'AGNdT9':
        sim_type = 'PERIODIC'

    return sim_type

def lum_write_out(tag, kappa, filters = FLARE.filters.TH[:-1], inp = 'FLARES'):

    Mlumintr = get_lum_all(0, tag, filters = filters, LF = False, inp = inp, Type = 'Pure-stellar')
    MlumBC = get_lum_all(0, tag, filters = filters, LF = False, inp = inp, Type = 'Only-BC')
    Mlumatt = get_lum_all(kappa, tag, filters = filters, LF = False, inp = inp, Type = 'Total')

    if inp == 'FLARES':
        df = pd.read_csv('weight_files/weights_grid.txt')
        weights = np.array(df['weights'])
        n = len(weights)
    else: n = 1

    for kk in range(n):

        if inp == 'FLARES':
            num = str(kk)
            if len(num) == 1:
                num =  '0'+num

            filename = F"./data/FLARES_{num}_sp_info.hdf5"
            sim_type = inp
            out_int = Mlumintr[kk]
            out_BC = MlumBC[kk]
            out_att = Mlumatt[kk]

        elif (inp == 'REF') or (inp == 'AGNdT9'):
            filename = F"./data/EAGLE_{inp}_sp_info.hdf5"
            sim_type = 'PERIODIC'
            out_int = Mlumintr
            out_BC = MlumBC
            out_att = Mlumatt

        else:
            ValueError(F"No input option of {inp}")

        fl = flares.flares(fname = filename, sim_type = sim_type)

        fl.create_group(F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic")
        fl.create_group(F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/No_ISM")
        fl.create_group(F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI")

        for ii, jj in enumerate(filters):

            filter = jj[8:]

            fl.create_dataset(values = out_int[:,ii], name = F"{filter}",
            group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic",
            desc = F'Intrinsic luminosity of the galaxy in the {filter} band', unit = "ergs/s/Hz")

            fl.create_dataset(values = out_BC[:,ii], name = F"{filter}",
            group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/No_ISM",
            desc = F'Intrinsic (stellar + nebular) luminosity of the galaxy with BC attenuation in the {filter} band', unit = "ergs/s/Hz")

            fl.create_dataset(values = out_att[:,ii], name = F"{filter}",
            group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI",
            desc = F"Dust corrected luminosity (using ModelI) of the galaxy in the {filter} band", unit = "ergs/s/Hz")


def flux_write_out(tag, kappa, filters = FLARE.filters.ACS, inp = 'FLARES'):

    Mfluxstell = get_flux_all(0, tag, filters = filters, inp = inp, Type = 'Pure-stellar')
    Mfluxintr = get_flux_all(0, tag, filters = filters, inp = inp, Type = 'Only-BC')
    Mfluxatt = get_flux_all(kappa, tag, filters = filters, inp = inp, Type = 'Total')

    if inp == 'FLARES':
        df = pd.read_csv('weight_files/weights_grid.txt')
        weights = np.array(df['weights'])
        n = len(weights)
    else: n = 1

    for kk in range(n):

        if inp == 'FLARES':
            num = str(kk)
            if len(num) == 1:
                num =  '0'+num

            filename = F"./data/FLARES_{num}_sp_info.hdf5"
            sim_type = inp
            out_stell = Mfluxstell[kk]
            out_int = Mfluxintr[kk]
            out_att = Mfluxatt[kk]

        elif (inp == 'REF') or (inp == 'AGNdT9'):
            filename = F"./data/EAGLE_{inp}_sp_info.hdf5"
            sim_type = 'PERIODIC'
            out_stell = Mfluxstell
            out_int = Mfluxintr
            out_att = Mfluxatt

        else:
            ValueError(F"No input option of {inp}")

        fl = flares.flares(fname = filename, sim_type = sim_type)
        fl.create_group(F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Flux/Intrinsic")
        fl.create_group(F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Flux/No_ISM")
        fl.create_group(F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Flux/DustModelI")

        for ii, jj in enumerate(filters):

            filter = re.findall('\w+', jj) #index `0` is the telescope, `1` is the name
            #of the instrument and `2` is the name of the filter

            fl.create_dataset(values = out_stell[:,ii], name = F"{filter[0]}/{filter[1]}/{filter[2]}",
            group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Flux/Intrinsic",
            desc = F"Intrinsic flux of the galaxy in the {filter}", unit = "nJy")

            fl.create_dataset(values = out_int[:,ii], name = F"{filter[0]}/{filter[1]}/{filter[2]}",
            group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Flux/No_ISM",
            desc = F"Intrinsic (stellar + nebular) flux of the galaxy with BC attenuation in the {filter}", unit = "nJy")

            fl.create_dataset(values = out_att[:,ii], name = F"{filter[0]}/{filter[1]}/{filter[2]}",
            group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Flux/DustModelI",
            desc = F"Dust corrected flux (using ModelI) of the galaxy in the {filter}", unit = "nJy")


if __name__ == "__main__":

    """

    There is no parallel hdf5 write at the moment. It is ok for now, as they don't
    seem to overlap. Would need to fix that with parallel_hdf5 later

    """

    #Filters for flux: ACS + WFC3NIR_W, IRAC, Euclid, Subaru, NIRCam

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    inp, prop = sys.argv[1], sys.argv[2]
    kappa =  0.0875

    sim_type = get_simtype(inp)
    fl = flares.flares(fname = 'tmp', sim_type = sim_type)
    tags = fl.tags


    for ii, tag in enumerate(tags):
        if (ii == rank) & (prop == 'Luminosity'):
            print(F'rank = {rank}, tag = {tag}')
            lum_write_out(tag = tag, kappa = kappa, inp = inp)

        if (ii == rank) & (prop == 'Flux'):
            print(F'rank = {rank}, tag = {tag}')
            flux_write_out(tag = tag, kappa = kappa, filters = FLARE.filters.ACS+FLARE.filters.WFC3NIR_W+FLARE.filters.IRAC+FLARE.filters.Euclid+FLARE.filters.Subaru+FLARE.filters.NIRCam, inp = inp)

        else:
            ValueError(F"No input of type {prop}")
