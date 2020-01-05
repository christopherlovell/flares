import numpy as np
import h5py
import sys

class HDF5_write(object):
    """
    Simple class to append value to a hdf5 file

    Params:
        datapath: filepath of h5 file
        group: group name within the file
        dataset: dataset name within the file or group
        value: dataset value, used to create and append to the dataset
        dtype: numpy dtype

    Usage:
        shape = (20,3,)
        hdf5_store = HDF5Store('hdf5_store.hdf5')
        hdf5_store.create_grp('Stars')
        hdf5_store.create_dset(np.random.random(shape), 'X', 'Stars')
        for _ in range(10):
            hdf5_store.append(np.random.random(shape), 'X', 'Stars')

    Modified from https://gist.github.com/wassname/a0a75f133831eed1113d052c67cf8633
    """
    def __init__(self, datapath, compression="gzip"):
        self.datapath = datapath
        self.compression = compression


    def create_grp(self, group, desc = 'None'):
        with h5py.File(self.datapath, mode='a') as h5f:
            if group not in list(h5f.keys()):
                dset = h5f.create_group(group)
                dset.attrs['Description'] = desc
            else:
                print("`{}` group already created".format(group))
                sys.exit()


    def create_header(self, hdr_name, value):
        with h5py.File(self.datapath, mode='a') as h5f:
            if hdr_name not in list(h5f.keys()):
                hdr = h5f.create_group(hdr_name)
                for ii in value.keys():
                    hdr.attrs[ii] = value[ii]
            else:
                print("`{}` attributes group already created".format(hdr_name))
                sys.exit()


    def create_dset(self, values, dataset, group = 'None', dtype=np.float64, desc = 'None', unit = 'None'):
        with h5py.File(self.datapath, mode='a') as h5f:
            shape = np.shape(values)
            if group == 'None':
                dset = h5f.create_dataset(
                dataset,
                shape=shape,
                maxshape=(None, ) + shape[1:],
                dtype=dtype,
                compression=self.compression,
                data=values)

                dset.attrs['Description'] = desc
                dset.attrs['Units'] = unit

            else:
                try:
                    dset = h5f[group].create_dataset(
                    dataset,
                    shape=shape,
                    maxshape=(None,) + shape[1:],
                    dtype=dtype,
                    compression=self.compression,
                    data=values)

                    dset.attrs['Description'] = desc
                    dset.attrs['Units'] = unit

                except Exception as e:
                    print("Oh! Oh! something went wrong while creating {}/{} or it already exists.\nNo value was written into the dataset.".format(group, dataset))
                    print (e)
                    sys.exit


    def append(self, values, dataset, group = 'None'):
        with h5py.File(self.datapath, mode='a') as h5f:
            if group == 'None':
                dset = h5f[dataset]
            else:
                dset = h5f["{}/{}".format(group, dataset)]
            ini = len(dset)
            if np.isscalar(values):
                add = 1
            else:
                add = len(values)
            dset.resize(ini+add, axis = 0)
            dset[ini:] = values
            h5f.flush()

    def write_attribute(self, loc, desc):
        with h5py.File(self.datapath, mode='a') as h5f:
            dset = h5f[loc]
            dset.attrs['Description'] = desc
            dset.close()


if __name__== "__main__":

    shape = (10,)
    hdf5_store = HDF5_write('hdf5_store.hdf5')

    hdf5_store.create_grp('Gas')
    hdf5_store.create_dset(np.random.random(shape), 'X', 'Gas', desc = 'gas particles', unit = 'Msun')

    for ii in range(10):

        hdf5_store.append(np.random.random(shape), 'X', 'Gas')
