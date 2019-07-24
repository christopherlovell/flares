
import numpy as np
import eagle as E
import h5py
import flares

from h5py_utilities import check_h5py, write_data_h5py, create_group_h5py, save_dict_to_hdf5, load_dict_from_hdf5


fl = flares.flares()
centre = [3200./2,3200./2,3200./2]
fname = 'data/flares_%d.h5'

if check_h5py(fname,'radius'):
    r = load_dict_from_hdf5(fname,'radius')
else:
    r = {h: {t: 0. for t in fl.tags} for h in fl.halos}
    save_dict_to_hdf5(dic=r,filename=fname,groupname='radius')


create_group_h5py(fname,'masks')

for halo in fl.halos:
    
    print("Halo: %s"%halo)
    halo_dir = fl.directory + 'geagle_' + halo + '/data/' 

    create_group_h5py(fname, 'masks/%s'%halo)

    if fl.check_snap_exists(halo, fl.tags[1]): tags = fl.tags
    else: tags = fl.alt_tags

    for rtag,tag in zip(fl.tags,tags):

        # print("Loading data...",rtag,tag)
        # dm_cood = E.readArray('SNAPSHOT', halo_dir, tag, "/PartType1/Coordinates", physicalUnits=False, noH=True, numThreads=1)
        cop = E.readArray("SUBFIND", halo_dir, tag, "/Subhalo/CentreOfPotential", numThreads=1, noH=True, physicalUnits=False)

        # r[halo][rtag] = fl.cut_radius(dm_cood[:,0],dm_cood[:,1],dm_cood[:,2],3200/2,3200/2,3200/2, threshold = 0.6, cut = 1.5)
        r[halo][rtag] = fl.cut_radius(cop[:,0],cop[:,1],cop[:,2],
                                      3200/2,3200/2,3200/2, 
                                      threshold = 0.6, cut = 1.)
   
        save_dict_to_hdf5(r,fname,'radius', overwrite=True)
        
        mask = fl.cutout(cop,centre,r[halo][rtag])
        
        gstr = 'masks/%s'%halo
        write_data_h5py(fname, gstr, rtag, data=mask, overwrite=True)
        print(rtag, r[halo][rtag])


