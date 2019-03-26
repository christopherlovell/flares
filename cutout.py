
import numpy as np

import eagle as E
import h5py
import flares



def check_h5py(hf_obj, obj_str):
    if obj_str not in hf_obj:
        return False
    else:
        return True

def write_data_h5py(hf_obj, obj_str, data, overwrite=False):
    
    if check_h5py(hf_obj, obj_str):
        if overwrite:
            print('Overwriting data in %s'%obj_str)
            old_data = hf_obj[obj_str]
            old_data[...] = data
        else:
            raise ValueError('Dataset already exists, and `overwrite` not set')
    else:
        hf_obj.create_dataset(obj_str, data=data)


def create_group_h5py(hf_obj, obj_str):
    if check_h5py(hf_obj, obj_str):
        print('Object already exists')
        return False
    else:
        hf_obj.create_group(obj_str)


fl = flares.flares()

centre = [3200./2,3200./2,3200./2]

r = {h: {t: 0. for t in fl.tags} for h in fl.halos}

hf = h5py.File('data/flares_%d.h5','a',driver='family')
create_group_h5py(hf,'masks')

for halo in fl.halos:
    
    print("Halo: %s"%halo)
    halo_dir = fl.directory + 'geagle_' + halo + '/data/' 

    create_group_h5py(hf, 'masks/%s'%halo)

    for tag in fl.tags:

        dm_cood = E.readArray('SNAPSHOT', halo_dir, tag, "/PartType1/Coordinates", physicalUnits=False, noH=True, numThreads=1)
        cop = E.readArray("SUBFIND", halo_dir, tag, "/Subhalo/CentreOfPotential", numThreads=1, noH=True, physicalUnits=False)

        r[halo][tag] = fl.cut_radius(dm_cood[:,0],dm_cood[:,1],dm_cood[:,2],3200/2,3200/2,3200/2, threshold = 0.6, cut = 1.5)

        mask = fl.cutout(cop,centre,r[halo][tag])

        hstr = 'masks/%s/%s'%(halo,tag)
        write_data_h5py(hf, hstr, data=mask, overwrite=True)



print("r:\n",[[d2 for k2,d2 in d1.items()] for k1,d1 in r.items()])

create_group_h5py(hf,'cutout')

radius_arr = np.array( [[d2 for k2,d2 in d1.items()] for k1,d1 in r.items()]).flatten()
tag_arr = np.array( [[k2 for k2,d2 in d1.items()] for k1,d1 in r.items()], dtype=np.string_).flatten()
halo_arr = np.array( [[k1 for k2,d2 in d1.items()] for k1,d1 in r.items()], dtype=np.string_).flatten()

write_data_h5py(hf,'cutout/radius',radius_arr,overwrite=True)
write_data_h5py(hf,'cutout/tag',tag_arr,overwrite=True)
write_data_h5py(hf,'cutout/halo',halo_arr,overwrite=True)

hf.close()

