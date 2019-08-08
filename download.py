import sys
import numpy as np

from eagle_IO import eagle_IO as E
import flares
from h5py_utilities import write_data_h5py, create_group_h5py, check_h5py

overwrite = False

fl = flares.flares()
fname = 'data/flares_%d.h5'

create_group_h5py(fname, 'mstar')
create_group_h5py(fname, 'sfr')
create_group_h5py(fname, 'centrals')

for halo in fl.halos:
    
    print(halo)

    create_group_h5py(fname, 'mstar/%s'%halo)
    create_group_h5py(fname, 'sfr/%s'%halo)
    create_group_h5py(fname, 'centrals/%s'%halo)

    if halo in ['0014']: overwrite = True
    else: overwrite = False


    for tag in fl.tags:

        halodir = fl.directory+'/geagle_'+halo+'/data/'

        if (check_h5py(fname, 'mstar/%s/%s'%(halo,tag)) is False) |\
                (overwrite == True):
            
            mstar = E.read_array("SUBFIND", halodir, tag, 
                                 "/Subhalo/Stars/Mass", 
                                 numThreads=1, noH=True)

            write_data_h5py(fname, 'mstar/%s'%halo, tag, 
                            data=mstar, overwrite=True)
           

        if (check_h5py(fname, 'sfr/%s/%s'%(halo,tag)) is False) |\
                (overwrite == True):
            
            sfr = E.read_array("SUBFIND", halodir, tag, 
                               "/Subhalo/StarFormationRate", 
                               numThreads=1, noH=True)

            write_data_h5py(fname, 'sfr/%s'%halo, tag, 
                            data=sfr, overwrite=True)


        if (check_h5py(fname, 'centrals/%s/%s'%(halo,tag)) is False) |\
                (overwrite == True):

            centrals = (E.read_array("SUBFIND", halodir, tag, 
                                     "/Subhalo/SubGroupNumber", 
                                     numThreads=1, noH=True) == 0)
            
            write_data_h5py(fname, 'centrals/%s'%halo, tag, 
                            data=centrals, overwrite=True)
            
