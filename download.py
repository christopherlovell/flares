import pickle as pcl
import sys

import numpy as np
import eagle as E

import flares
from h5py_utilities import write_data_h5py, create_group_h5py

fl = flares.flares()
fname = 'data/flares_%d.h5'

create_group_h5py(fname, 'mstar')
create_group_h5py(fname, 'sfr')
# create_group_h5py(fname, 'coods')

for halo in fl.halos:

    create_group_h5py(fname, 'mstar/%s'%halo)
    create_group_h5py(fname, 'sfr/%s'%halo)

    # if fl.check_snap_exists(halo, fl.tags[1]): tags = fl.tags
    # else: tags = fl.alt_tags
    tags = fl.tags

    for tag,rtag in zip(tags,fl.tags):

        print(halo, tag)
        halodir = fl.directory+'/geagle_'+halo+'/data/'

        mstar = E.readArray("SUBFIND", halodir, tag, "/Subhalo/Stars/Mass", numThreads=1, noH=True)
        sfr = E.readArray("SUBFIND", halodir, tag, "/Subhalo/StarFormationRate", numThreads=1, noH=True)

        write_data_h5py(fname, 'mstar/%s'%halo, rtag, data=mstar, overwrite=True)
        write_data_h5py(fname, 'sfr/%s'%halo, rtag, data=sfr, overwrite=True)


        # coods[halo][tag] = E.readArray("SUBFIND", halodir, tag, "/Subhalo/CentreOfPotential", numThreads=1, noH=True, physicalUnits=False)
        # bhmass[halo][tag] = E.readArray("SUBFIND", halodir, tag, "/Subhalo/BlackHoleMass", numThreads=1, noH=True)
        # tmass[halo][tag] = E.readArray("SUBFIND", halodir, tag, "/Subhalo/Mass", numThreads=1, noH=True)
        # Zstars[halo][tag] = E.readArray("SUBFIND", halodir, tag, "/Subhalo/Stars/Metallicity", numThreads=1, noH=True)
        # ZgasSF[halo][tag] = E.readArray("SUBFIND", halodir, tag, "/Subhalo/SF/Metallicity", numThreads=1, noH=True)
        # grpn = E.readArray("SUBFIND", halodir, tag, "/Subhalo/GroupNumber", numThreads=1)
        # m200[halo][tag] = E.readArray("SUBFIND_GROUP", halodir, tag, "/FOF/Group_M_Mean200", numThreads=1, noH=True)[grpn - 1]
    #     bhaccr[halo][tag] = E.readArray("SUBFIND", halodir, tag, "/Subhalo/BlackHoleMassAccretionRate", numThreads=1, noH=True)
    #     centrals[halo][tag] = (E.readArray("SUBFIND", halodir, tag, "/Subhalo/SubGroupNumber", numThreads=1, noH=True) == 0)
    #     ZgasNSF[halo][tag] = E.readArray("SUBFIND", halodir, tag, "/Subhalo/NSF/Metallicity", numThreads=1, noH=True)

