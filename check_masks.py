"""
Check arrays are the same size. 

Utility script
"""

import flares
fl = flares.flares()

## Load GEAGLE ## 
fname = fl.data+'/flares.h5'
mstar = fl.load_dict_from_hdf5(fname, 'mstar')
R = fl.load_dict_from_hdf5(fname, 'radius')
vmask = fl.load_dict_from_hdf5(fname, 'masks')

for tag in fl.tags:
    for halo in fl.halos:
        if mstar[halo][tag].shape != vmask[halo][tag].shape:
            print("Arrays not equal sized!\nmstar | vmask\n%s %s\n"%(halo,tag))
            print(mstar[halo][tag].shape, vmask[halo][tag].shape)

