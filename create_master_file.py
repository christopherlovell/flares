import h5py
import flares

fl = flares.flares('./data/flares.hdf5',sim_type='FLARES')

in_dir = './data/'

with h5py.File('./data/flares.hdf5','w') as outfile:

    for halo in fl.halos:
        print(halo)

        fl.create_group(F'{halo}')

        infile = h5py.File('%s/FLARES_%s_sp_info.hdf5'%(in_dir,halo),'r')

        for tag in fl.tags:
            infile.copy(tag,outfile[halo])
