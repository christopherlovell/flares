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


# if __name__ == '__main__':
# 
#     data = {'x': 'astring',
#             'y': np.arange(10),
#             'd': {'z': np.ones((2,3)),
#                   'b': b'bytestring'}}
#     print(data)
#     filename = 'test.h5'
#     save_dict_to_hdf5(data, filename)
#     dd = load_dict_from_hdf5(filename)
#     print(dd)
#     # should test for bad type


