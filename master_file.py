import h5py
import flares
from HDF5_write import HDF5_write


fl = flares.flares()

outfile = h5py.File('data/flares.hdf5','w')

for halo in fl.halos:
    print(halo)

    fl.create_group_h5py('data/flares.hdf5',halo)

    infile = h5py.File('data/GEAGLE_%s_sp_info.hdf5'%halo,'r')

    infile.copy('010_z005p000',outfile[halo])


    #
    # recursively_save_contents(outfile, halo, infile)
