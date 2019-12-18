import h5py
import flares

fl = flares.flares()

outfile = h5py.File('data/flares.h5','w')

for halo in fl.halos:
    print(halo)
    
    fl.create_group_h5py('data/flares.h5',halo)
    
    infile = h5py.File('data/GEAGLE_%s_sp_info.hdf5'%halo,'r') 
   
    for tag in fl.tags:
        infile.copy(tag,outfile[halo])

