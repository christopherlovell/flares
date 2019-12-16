import h5py

import flares

from HDF5_write import HDF5_write


def recursively_save_contents(self, h5file_A, path, h5file_B, overwrite=False):
    """
    ....
    """
    for key,item in h5file_B.items():
        #if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes, float)):
        if isinstance(item,h5py.dataset):
            if overwrite:
                old_data = h5file[path + key]
                old_data[...] = item
            else:
                h5file[path + key] = item
        elif isinstance(item, h5py.Group):
            self.recursively_save_dict_contents_to_group(h5file, path + key + '/', item, overwrite=overwrite)
        else:
            raise ValueError('Cannot save %s type'%type(item))



fl = flares.flares()

outfile = h5py.File('data/flares.h5','w')

for halo in ['38']:#fl.halos:
    print(halo)
    
    fl.create_group_h5py('data/flares.h5',halo)
    
    infile = h5py.File('data/GEAGLE_%s_sp_info.hdf5'%halo,'r') 
    
    infile.copy('010_z005p000',outfile[halo])


    # 
    # recursively_save_contents(outfile, halo, infile)






