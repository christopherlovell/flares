
import numpy as np
import eagle as E
import h5py
import flares

from h5py_utilities import check_h5py, write_data_h5py, create_group_h5py, load_h5py

overwrite = False
verbose = True

fl = flares.flares()
centre = [3200./2,3200./2,3200./2]
fname = 'data/flares_%d.h5'

# if False: #check_h5py(fname,'radius'):
#     r = load_dict_from_hdf5(fname,'radius')
# else:
# with h5py.File('data/flares_%d.h5','a',driver='family') as f:
#     del f['radius']


# r = {h: {t: 0. for t in fl.tags} for h in fl.halos}
# fl.save_dict_to_hdf5(dic=r,filename=fname,groupname='radius')


create_group_h5py(fname,'radius')
create_group_h5py(fname,'masks')

for halo in fl.halos:
    
    if verbose: print("Halo: %s"%halo)
    halo_dir = fl.directory + 'geagle_' + halo + '/data/' 

    create_group_h5py(fname, 'masks/%s'%halo)
    create_group_h5py(fname, 'radius/%s'%halo)

    # if fl.check_snap_exists(halo, fl.tags[1]): tags = fl.tags
    # else: tags = fl.alt_tags
    tags = fl.tags

    for rtag,tag in zip(fl.tags,tags):
        if verbose: print("Loading data...",rtag)

        if (check_h5py(fname, 'radius/%s/%s'%(halo,tag)) is False) |\
            (overwrite is True):
        
            cop = E.readArray("SUBFIND", halo_dir, tag, 
                              "/Subhalo/CentreOfPotential", numThreads=1, 
                              noH=True, physicalUnits=False)
    
            r = fl.cut_radius(cop[:,0],cop[:,1],cop[:,2],
                              3200/2,3200/2,3200/2, 
                              threshold = 0.6, cut = 1.)
   
            gstr = 'radius/%s'%halo
            write_data_h5py(fname,gstr,rtag,data=r, overwrite=True)
            
            mask = fl.cutout(cop,centre,r)
            
            gstr = 'masks/%s'%halo
            write_data_h5py(fname,gstr,rtag,data=mask, overwrite=True)
            
            print(rtag, r)

        else:
            r = load_h5py(fname, 'radius/%s/%s'%(halo,tag))
            if verbose: print('Already exists (r=%d), overwrite=False'%r)
        
