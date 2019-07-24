import h5py

def load_h5py(filename, obj_str):
    with h5py.File(filename, 'a', driver='family') as h5file:
        dat = np.array(h5file.get(obj_str))
    return dat

def check_h5py(filename, obj_str):
    with h5py.File(filename, 'a', driver='family') as h5file:
        if obj_str not in h5file:
            return False
        else:
            return True

def write_data_h5py(filename, grp_str, name, data, overwrite=False):
    full_str = "%s/%s"%(grp_str,name)
    check = check_h5py(filename, full_str)
    
    with h5py.File(filename, 'a', driver='family') as h5file:
        if check:
            if overwrite:
                print('Overwriting data in %s'%full_str)
                old_data = h5file[full_str]
                old_data[...] = data
            else:
                raise ValueError('Dataset already exists, and `overwrite` not set')
        else:
            grp = h5file[grp_str]
            grp.create_dataset(name, data=data)


def create_group_h5py(filename, obj_str):
    check = check_h5py(filename, obj_str)
    with h5py.File(filename, 'a', driver='family') as h5file:
        if check:
            print('Object already exists')
            return False
        else:
            h5file.create_group(obj_str)



"""
Utilities for loading and saving nested dictionaries recursively

see https://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py
"""
import numpy as np
import os

def save_dict_to_hdf5(dic, filename, groupname='default', overwrite=False):
    """
    ....
    """
    if groupname=='default': print("Saving to `default` group")
    with h5py.File(filename, 'a', driver='family') as h5file:
        recursively_save_dict_contents_to_group(h5file, groupname+'/', dic, overwrite=overwrite)

def recursively_save_dict_contents_to_group(h5file, path, dic, overwrite=False):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes, float)):
            if overwrite:
                old_data = h5file[path + key]
                old_data[...] = item
            else:
                h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item, overwrite=overwrite)
        else:
            raise ValueError('Cannot save %s type'%type(item))

def load_dict_from_hdf5(filename, group=''):
    """
    ....
    """
    with h5py.File(filename, 'r', driver='family') as h5file:
        return recursively_load_dict_contents_from_group(h5file, group+'/')

def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()] # .value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

if __name__ == '__main__':

    data = {'x': 'astring',
            'y': np.arange(10),
            'd': {'z': np.ones((2,3)),
                  'b': b'bytestring'}}
    print(data)
    filename = 'test.h5'
    save_dict_to_hdf5(data, filename)
    dd = load_dict_from_hdf5(filename)
    print(dd)
    # should test for bad type


