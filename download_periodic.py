import sys

import numpy as np
import eagle as E

import flares
from h5py_utilities import write_data_h5py, create_group_h5py, check_h5py

fl = flares.flares()
fname = 'data/periodic_%d.h5'


for box in ['ref','agn']:
    create_group_h5py(fname, box)
    create_group_h5py(fname, '%s/mstar'%box)
    create_group_h5py(fname, '%s/sfr'%box)
    create_group_h5py(fname, '%s/centrals'%box)


for tag in fl.ref_tags:
    print(tag)

    ## Reference ##
    if check_h5py(fname, 'ref/mstar/%s'%tag) is False:
        mstar = E.readArray("SUBFIND", fl.ref_directory, tag, "/Subhalo/Stars/Mass", 
                            numThreads=1, noH=True)
        write_data_h5py(fname, 'ref/mstar', tag, data=mstar, overwrite=True)


    if check_h5py(fname, 'ref/sfr/%s'%tag) is False:
        sfr = E.readArray("SUBFIND", fl.ref_directory, tag, "/Subhalo/StarFormationRate", 
                          numThreads=1, noH=True)
        write_data_h5py(fname, 'ref/sfr', tag, data=sfr, overwrite=True)


    if check_h5py(fname, 'ref/centrals/%s'%tag) is False:
        centrals = (E.readArray("SUBFIND", fl.ref_directory, tag, "/Subhalo/SubGroupNumber", 
                                numThreads=1, noH=True) == 0)
        write_data_h5py(fname, 'ref/centrals', tag, data=centrals, overwrite=True)


    ## AGNdT9 ## 
    if check_h5py(fname, 'agn/mstar/%s'%tag) is False:
        mstar = E.readArray("SUBFIND", fl.agn_directory, tag, "/Subhalo/Stars/Mass", 
                            numThreads=1, noH=True)
        write_data_h5py(fname, 'agn/mstar', tag, data=mstar, overwrite=True)
        
        
    if check_h5py(fname, 'agn/sfr/%s'%tag) is False:
        sfr = E.readArray("SUBFIND", fl.agn_directory, tag, "/Subhalo/StarFormationRate", 
                          numThreads=1, noH=True)
        write_data_h5py(fname, 'agn/sfr', tag, data=sfr, overwrite=True)


    if check_h5py(fname, 'agn/centrals/%s'%tag) is False:
        centrals = (E.readArray("SUBFIND", fl.agn_directory, tag, "/Subhalo/SubGroupNumber", 
                                numThreads=1, noH=True) == 0)
        write_data_h5py(fname, 'agn/centrals', tag, data=centrals, overwrite=True)






    # coods[halo][tag] = E.readArray("SUBFIND", halodir, tag, "/Subhalo/CentreOfPotential", numThreads=1, noH=True, physicalUnits=False)
    # bhmass[halo][tag] = E.readArray("SUBFIND", halodir, tag, "/Subhalo/BlackHoleMass", numThreads=1, noH=True)
    # tmass[halo][tag] = E.readArray("SUBFIND", halodir, tag, "/Subhalo/Mass", numThreads=1, noH=True)
    # Zstars[halo][tag] = E.readArray("SUBFIND", halodir, tag, "/Subhalo/Stars/Metallicity", numThreads=1, noH=True)
    # ZgasSF[halo][tag] = E.readArray("SUBFIND", halodir, tag, "/Subhalo/SF/Metallicity", numThreads=1, noH=True)
    # grpn = E.readArray("SUBFIND", halodir, tag, "/Subhalo/GroupNumber", numThreads=1)
    # m200[halo][tag] = E.readArray("SUBFIND_GROUP", halodir, tag, "/FOF/Group_M_Mean200", numThreads=1, noH=True)[grpn - 1]
#     bhaccr[halo][tag] = E.readArray("SUBFIND", halodir, tag, "/Subhalo/BlackHoleMassAccretionRate", numThreads=1, noH=True)

#     ZgasNSF[halo][tag] = E.readArray("SUBFIND", halodir, tag, "/Subhalo/NSF/Metallicity", numThreads=1, noH=True)

