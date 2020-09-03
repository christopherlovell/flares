import os
import numpy as np
from astropy.cosmology import Planck13 as cosmo
from astropy import units as u
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
import scipy.stats
import h5py
import schwimmbad
from functools import partial
import eagle_IO.eagle_IO as E
from numba import jit, njit

norm = np.linalg.norm
conv = (u.solMass/u.Mpc**2).to(u.solMass/u.pc**2)



class flares:

    def __init__(self,fname,sim_type='FLARES'):

        self.fname =fname
        self.compression = 'gzip'
        self.sim_type = sim_type

        #Put down the sim root location here
        #self.directory = '/cosma7/data/dp004/dc-payy1/G-EAGLE/GEAGLE_'
        self.directory = '/cosma7/data/dp004/dc-payy1/G-EAGLE/'
        self.ref_directory = '/cosma7/data//Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data'
        self.agn_directory = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0050N0752/PE/S15_AGNdT9/data'

        self.cosmo = cosmo

        # self.data = "%s/data"%(os.path.dirname(os.path.realpath(__file__)))

        if self.sim_type == 'FLARES':
            self.radius = 14. #in cMpc/h
            self.halos = np.array([
                     '00','01','02','03','04',
                     '05','06','07','08','09',
                     '10','11','12','13','14',
                     '15','16','17','18','19',
                     '20','21','22','23','24',
                     '25','26','27','28','29',
                     '30','31','32','33','34',
                     '35','36','37','38','39'])

            self.tags = np.array([
                                  #'000_z015p000','001_z014p000','002_z013p000',
                                  #'003_z012p000','004_z011p000',
                                  '005_z010p000',
                                  '006_z009p000','007_z008p000','008_z007p000',
                                  '009_z006p000','010_z005p000'
                                  #,'011_z004p770'
                                  ])

            self.zeds = [float(tag[5:].replace('p','.')) for tag in self.tags]

            ## update with weights file location
            self.weights = '/cosma7/data/dp004/dc-love2/codes/ic_selection/weights_grid.txt'
        elif sim_type=="PERIODIC":
            self.tags = np.array(['002_z009p993','003_z008p988',
                                  '004_z008p075','005_z007p050','006_z005p971',
                                  '008_z005p037'])
            self.zeds = [float(tag[5:].replace('p','.')) for tag in self.tags]
        else:
            raise ValueError("sim_type not recognised")



    def print_keys(self, obj='/'):
        with h5py.File(self.fname,'r') as f:
           if isinstance(f[obj], h5py.Dataset):
               print("Dataset, no keys.")
           else:
               print(list(f[obj].keys()))


    def print_attributes(self,obj):
        with h5py.File(self.fname,'r') as f: 
            _attrs = list(f[obj].attrs.keys())
            if len(_attrs) < 1:
                print("No attributes.")
            else:
                for key in _attrs:
                    print("%s:"%key, f[obj].attrs[key])     


    # @jit
    def _sphere(self, coords, a, b, c, r):

        #Equation of a sphere

        x, y, z = coords[:,0], coords[:,1], coords[:,2]

        return (x-a)**2 + (y-b)**2 + (z-c)**2 - r**2


    def spherical_region(self, sim, snap):

        """
        Inspired from David Turner's suggestion
        """

        dm_cood = E.read_array('PARTDATA', sim, snap, '/PartType1/Coordinates',
        noH=False, physicalUnits=False, numThreads=4)  #dm particle coordinates

        hull = ConvexHull(dm_cood)

        cen = [np.median(dm_cood[:,0]), np.median(dm_cood[:,1]), np.median(dm_cood[:,2])]
        pedge = dm_cood[hull.vertices]  #edge particles
        y_obs = np.zeros(len(pedge))
        p0 = np.append(cen, self.radius)

        popt, pcov = curve_fit(self._sphere, pedge, y_obs, p0, method = 'lm', sigma = np.ones(len(pedge))*0.001)
        dist = np.sqrt(np.sum((pedge-popt[:3])**2, axis = 1))
        centre, radius, mindist = popt[:3], popt[3], np.min(dist)

        return centre, radius, mindist


    def calc_df(self, mstar, tag, volume, massBinLimits):

        hist, dummy = np.histogram(np.log10(mstar), bins = massBinLimits)
        hist = np.float64(hist)
        phi = (hist / volume) / (massBinLimits[1] - massBinLimits[0])

        # p = 0.95 
        # phi_sigma = np.array([scipy.stats.chi2.ppf((1.-p)/2.,2*hist)/2.,
        #                       scipy.stats.chi2.ppf(p+(1.-p)/2.,2*(hist+1))/2.])

        # phi_sigma = (phi_sigma / volume) / (massBinLimits[1] - massBinLimits[0])

        phi_sigma = (np.sqrt(hist) / volume) /\
                    (massBinLimits[1] - massBinLimits[0]) # Poisson errors

        return phi, phi_sigma, hist


    def plot_df(self, ax, phi, phi_sigma, hist, massBins,
                label, color, hist_lim=10, lw=3, alpha=0.7, lines=True):

        kwargs = {}
        kwargs_lo = {}

        if lines:
            kwargs['lw']=lw

            kwargs_lo['lw']=lw
            kwargs_lo['linestyle']='dotted'
        else:
            kwargs['ls']=''
            kwargs['marker']='o'

            kwargs_lo['ls']=''
            kwargs_lo['marker']='o'
            kwargs_lo['markerfacecolor']='white'
            kwargs_lo['markeredgecolor']=color


        def yerr(phi,phi_sigma):

            p = phi
            ps = phi_sigma

            mask = (ps == p)

            err_up = np.abs(np.log10(p) - np.log10(p + ps))
            err_lo = np.abs(np.log10(p) - np.log10(p - ps))

            err_lo[mask] = 100

            return err_up, err_lo, mask


        err_up, err_lo, mask = yerr(phi,phi_sigma)
        
        # err_lo = np.log10(phi) - np.log10(phi - phi_sigma[0])
        # err_up = np.log10(phi) - np.log10(phi + phi_sigma[1])

        ax.errorbar(np.log10(massBins[phi > 0.]),
                np.log10(phi[phi > 0.]),
                yerr=[err_lo[phi > 0.],
                      err_up[phi > 0.]],
                #uplims=(mask[phi > 0.]),
                label=label, c=color, alpha=alpha, **kwargs)


    """
    Age of stellar particles
    """

    def get_star_formation_time(self, SFT, redshift):

        SFz = (1/SFT) - 1.
        SFz = self.cosmo.age(redshift).value - self.cosmo.age(SFz).value
        return SFz

    def get_age(self, arr, z, numThreads = 4):

        if numThreads == 1:
            pool = schwimmbad.SerialPool()
        elif numThreads == -1:
            pool = schwimmbad.MultiPool()
        else:
            pool = schwimmbad.MultiPool(processes=numThreads)

        calc = partial(self.get_star_formation_time, redshift = z)
        Age = np.array(list(pool.map(calc,arr)))

        return Age



    """
    HDF5 functionality
    """

    def _check_hdf5(self, obj_str):
        with h5py.File(self.fname, 'a') as h5file:
            if obj_str not in h5file:
                return False
            else:
                return True

    def create_group(self, group_name, desc=None, verbose=False):
        check = self._check_hdf5(group_name)
        if check:
            if verbose: print("`{}` group already created".format(group_name))
            return False

        with h5py.File(self.fname, 'a') as h5file:
            dset = h5file.create_group(group_name)
            if desc is not None:
                dset.attrs['Description'] = desc


    def create_header(self, hdr_name, value):
        with h5py.File(self.fname, mode='a') as h5f:
            if hdr_name not in list(h5f.keys()):
                hdr = h5f.create_group(hdr_name)
                for ii in value.keys():
                    hdr.attrs[ii] = value[ii]
            else:
                print("`{}` attributes group already created".format(hdr_name))
                return False
                #sys.exit()



    def create_dataset(self, values, name, group='/', overwrite=False,
                       dtype=np.float64, desc = None, unit = None, verbose=False):

        shape = np.shape(values)

        if self._check_hdf5(group) is False:
            raise ValueError("Group does not exist")
            return False

        try:
            with h5py.File(self.fname, mode='a') as h5f:

                if overwrite:
                    if verbose: print('Overwriting data in %s/%s'%(group,name))
                    if self._check_hdf5("%s/%s"%(group,name)) is True:
                        grp = h5f[group]
                        del grp[name]


                dset = h5f.create_dataset("%s/%s"%(group,name), shape=shape,
                                           maxshape=(None,) + shape[1:],
                                           dtype=dtype, compression=self.compression,
                                           data=values)

                if desc is not None:
                    dset.attrs['Description'] = desc
                if unit is not None:
                    dset.attrs['Units'] = unit

        except Exception as e:
            print("Oh! something went wrong while creating {}/{} or it already exists.\
                   \nNo value was written into the dataset.".format(group, name))
            print (e)
            # sys.exit


    def load_dataset(self,name,arr_type='Galaxy'):
        """ 
        Load a dataset for *all* halos and tags
        """

        if self.sim_type == "FLARES":
            out = {halo: {tag: None for tag in self.tags} for halo in self.halos}

            with h5py.File(self.fname,'r') as f:
                for halo in self.halos:
                    for tag in self.tags:
                        out[halo][tag] = f['%s/%s/%s/%s'%(halo,tag,arr_type,name)][:]
        elif self.sim_type == "PERIODIC":
            out = {tag: None for tag in self.tags}

            with h5py.File(self.fname,'r') as f:
                for tag in self.tags:
                    print('%s/%s/%s'%(tag,arr_type,name))
                    out[tag] = f['%s/%s/%s'%(tag,arr_type,name)][:]

        return out


    def _load_single_dataset(self,name,halo,tag,arr_type='Galaxy',verbose=False):
        """
        Identical to `load_dataset` but for a single halo/tag
        """

        if self.sim_type == "FLARES":
            # out = {halo: {tag: None for tag in self.tags} for halo in self.halos}

            with h5py.File(self.fname,'r') as f:
                out = f['%s/%s/%s/%s'%(halo,tag,arr_type,name)][:]
        elif self.sim_type == "PERIODIC":

            with h5py.File(self.fname,'r') as f:
                if verbose: print('%s/%s/%s'%(tag,arr_type,name))
                out = f['%s/%s/%s'%(tag,arr_type,name)][:]

        return out


    def append_to_dataset(self, values, dataset, group = '/'):
        with h5py.File(self.fname, mode='a') as h5f:
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
        with h5py.File(self.fname, mode='a') as h5f:
            dset = h5f[loc]
            dset.attrs['Description'] = desc
            #dset.close()


    def get_particles(self, p_str='S_Age',halo='00',tag='005_z010p000', verbose=False):
        """
        Grab particle properties for the given halo/tag and the given datasets 

        Args:
            p_str (str or list) particle dataset string, or a list / tuple of these strings
            halo (str)
            tag (str)
            verbose (bool)

        Returns:
            out (dict)  
        """
        if verbose: print("Getting particle array lengths...")
        # get particle array lengths for each galaxy
        S_length = self.load_dataset('S_Length',arr_type='Galaxy')
        # subset halo/tag
        S_len = S_length[halo][tag]
    
        if verbose: print("Finding array indices...")
        # find beginning:end indexes for each galaxy
        begin = np.zeros(len(S_len), dtype = np.int64)
        end = np.zeros(len(S_len), dtype = np.int64)
        begin[1:] = np.cumsum(S_len)[:-1]
        end = np.cumsum(S_len)
    
        if verbose: print("Getting particle data...")
        _p = {}
        if type(p_str) in [list,tuple]:
            for _str in p_str:
                _p[_str] = self._load_single_dataset(_str,halo,tag,arr_type='Particle')
        else:
            _p[p_str] = self._load_single_dataset(p_str,halo,tag,arr_type='Particle')
    
    
        if verbose: print("Subsetting particles for each galaxy...")
        # output dictionary of particle properties
        out = {}
        for i in np.arange(len(S_len)): # loop through gals
            out[i] = {}
    
            # if more than one property, loop through them
            if type(p_str) in [list,tuple]:
                for _str in p_str:
                    out[i][_str] = _p[_str][begin[i]:end[i]]
            else:
                out[i][p_str] = _p[p_str][begin[i]:end[i]]
    
    
        return out



    @staticmethod
    def _get_part_inds(halo_ids, part_ids, group_part_ids, sorted):
        """ A function to find the indexes and halo IDs associated to particles/a particle producing an array for each

        :param halo_ids:
        :param part_ids:
        :param group_part_ids:
        :return:
        """

        # Sort particle IDs if required and store an unsorted version in an array
        if sorted:
            part_ids = np.sort(part_ids)
        unsort_part_ids = np.copy(part_ids)

        # Get the indices that would sort the array (if the array is sorted this is just a range form 0-Npart)
        if sorted:
            sinds = np.arange(part_ids.size)
        else:
            sinds = np.argsort(part_ids)
            part_ids = part_ids[sinds]

        # Get the index of particles in the snapshot array from the in particles in a group array
        sorted_index = np.searchsorted(part_ids, group_part_ids)  # find the indices that would sort the array
        yindex = np.take(sinds, sorted_index, mode="raise")  # take the indices at the indices found above
        mask = unsort_part_ids[yindex] != group_part_ids  # define the mask based on these particles
        result = np.ma.array(yindex, mask=mask)  # create a mask array

        # Apply the mask to the id arrays to get halo ids and the particle indices
        part_groups = halo_ids[np.logical_not(result.mask)]  # halo ids
        parts_in_groups = result.data[np.logical_not(result.mask)]  # particle indices

        return parts_in_groups, part_groups

    def get_group_part_inds(self, sim, snapshot, part_type, all_parts=False, sorted=False):
        ''' A function to efficiently produce a dictionary of particle indexes from EAGLE particle data arrays
            for SUBFIND groups.

        :param sim:        Path to the snapshot file [str]
        :param snapshot:   Snapshot identifier [str]
        :param part_type:  The integer representing the particle type
                           (0, 1, 4, 5: gas, dark matter, stars, black hole) [int]
        :param all_parts:  Flag for whether to use all particles (SNAP group)
                           or only particles in halos (PARTDATA group)  [bool]
        :param sorted:     Flag for whether to produce indices in a sorted particle ID array
                           or unsorted (order they are stored in) [bool]
        :return:
        '''

        # Get the particle IDs for this particle type using eagle_IO
        if all_parts:

            # Get all particles in the simulation
            part_ids = E.read_array('SNAP', sim, snapshot, 'PartType' + str(part_type) + '/ParticleIDs',
                                    numThreads=8)

            # Get only those particles in a halo
            group_part_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/ParticleIDs',
                                          numThreads=8)

        else:

            # Get only those particles in a halo
            part_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/ParticleIDs',
                                    numThreads=8)

            # A copy of this array is needed for the extraction method
            group_part_ids = np.copy(part_ids)

        # Extract the group ID and subgroup ID each particle is contained within
        grp_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/GroupNumber',
                               numThreads=8)

        # Remove particles in unbound groups (groupnumber < 0)
        okinds = grp_ids < 0
        group_part_ids = group_part_ids[okinds]
        grp_ids = grp_ids[okinds]

        parts_in_groups, part_groups = self._get_part_inds(grp_ids, part_ids, group_part_ids, sorted)

        # Produce a dictionary containing the index of particles in each halo
        halo_part_inds = {}
        for ind, grp in zip(parts_in_groups, part_groups):
            halo_part_inds.setdefault(grp, set()).update({ind})

        # Now the dictionary is fully populated convert values from sets to arrays for indexing
        for key, val in halo_part_inds.items():
            halo_part_inds[key] = np.array(list(val))

        return halo_part_inds

    def get_subgroup_part_inds(self, sim, snapshot, part_type, all_parts=False, sorted=False):
        ''' A function to efficiently produce a dictionary of particle indexes from EAGLE particle data arrays
            for SUBFIND subgroups.

        :param sim:        Path to the snapshot file [str]
        :param snapshot:   Snapshot identifier [str]
        :param part_type:  The integer representing the particle type
                           (0, 1, 4, 5: gas, dark matter, stars, black hole) [int]
        :param all_parts:  Flag for whether to use all particles (SNAP group)
                           or only particles in halos (PARTDATA group)  [bool]
        :param sorted:     Flag for whether to produce indices in a sorted particle ID array
                           or unsorted (order they are stored in) [bool]
        :return:
        '''

        # Get the particle IDs for this particle type using eagle_IO
        if all_parts:

            # Get all particles in the simulation
            part_ids = E.read_array('SNAP', sim, snapshot, 'PartType' + str(part_type) + '/ParticleIDs',
                                    numThreads=8)

            # Get only those particles in a halo
            group_part_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/ParticleIDs',
                                          numThreads=8)

        else:

            # Get only those particles in a halo
            part_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/ParticleIDs',
                                    numThreads=8)

            # A copy of this array is needed for the extraction method
            group_part_ids = np.copy(part_ids)

        # Extract the group ID and subgroup ID each particle is contained within
        grp_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/GroupNumber',
                               numThreads=8)
        subgrp_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/SubGroupNumber',
                                  numThreads=8)

        # Remove particles not associated to a subgroup (subgroupnumber == 2**30 == 1073741824)
        okinds = subgrp_ids != 1073741824
        group_part_ids = group_part_ids[okinds]
        grp_ids = grp_ids[okinds]
        subgrp_ids = subgrp_ids[okinds]

        # Ensure no subgroup ID exceeds 99999
        assert subgrp_ids.max() < 99999, "Found too many subgroups, need to increase subgroup format string above %05d"

        # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
        halo_ids = np.zeros(grp_ids.size, dtype=float)
        for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
            halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

        parts_in_groups, part_groups = self._get_part_inds(halo_ids, part_ids, group_part_ids, sorted)

        # Produce a dictionary containing the index of particles in each halo
        halo_part_inds = {}
        for ind, grp in zip(parts_in_groups, part_groups):
            halo_part_inds.setdefault(grp, set()).update({ind})

        # Now the dictionary is fully populated convert values from sets to arrays for indexing
        for key, val in halo_part_inds.items():
            halo_part_inds[key] = np.array(list(val))

        return halo_part_inds

    def get_single_group_part_inds(self, halo_id, sim, snapshot, part_type, all_parts=False, sorted=False):
        ''' A wrapper function produce an array of particle indexes for a single group. NOTE: This will be VERY
            inefficient if done for many halos individually, better to create a dictionary using one of the other
            methods.

        :param halo_id:    The group number of the halo for which particle indexes are desired [int]
        :return:
        '''

        # Get the particle index dictionary
        halo_part_inds = self.get_group_part_inds(sim, snapshot, part_type, all_parts, sorted)

        # Extract the desired halo
        parts_in_group = halo_part_inds[halo_id]

        return parts_in_group

    def get_single_subgroup_part_inds(self, subhalo_id, sim, snapshot, part_type, all_parts=False, sorted=False):
        ''' A wrapper function produce an array of particle indexes for a single subgroup. NOTE: This will be VERY
            inefficient if done for many halos individually, better to create a dictionary using one of the other
            methods.

        :param halo_id:    The subgroup number of the halo for which particle indexes are desired [float]
                           NOTE: This must be of the form float(groupNumber.SubGroupNumber) format,
                           i.e. group 1 subgroup 11 = 1.00011
        :return:
        '''

        # Get the particle index dictionary
        halo_part_inds = self.get_group_part_inds(sim, snapshot, part_type, all_parts, sorted)

        # Extract the desired subhalo
        parts_in_group = halo_part_inds[subhalo_id]

        return parts_in_group


@jit()
def get_Z_LOS(s_cood, g_cood, g_mass, g_Z, g_sml, lkernel, kbins):

    """

    Compute the los metal surface density (in g/cm^2) for star particles inside the galaxy taking
    the z-axis as the los.
    Args:
        s_cood (3d array): stellar particle coordinates
        g_cood (3d array): gas particle coordinates
        g_mass (1d array): gas particle mass
        g_Z (1d array): gas particle metallicity
        g_sml (1d array): gas particle smoothing length

    """
    n = len(s_cood)
    Z_los_SD = np.zeros(n)
    #Fixing the observer direction as z-axis. Use make_faceon() for changing the
    #particle orientation to face-on
    xdir, ydir, zdir = 0, 1, 2
    for ii in range(n):

        thisspos = s_cood[ii]
        ok = (g_cood[:,zdir] > thisspos[zdir])
        thisgpos = g_cood[ok]
        thisgsml = g_sml[ok]
        thisgZ = g_Z[ok]
        thisgmass = g_mass[ok]
        x = thisgpos[:,xdir] - thisspos[xdir]
        y = thisgpos[:,ydir] - thisspos[ydir]

        b = np.sqrt(x*x + y*y)
        boverh = b/thisgsml

        ok = (boverh <= 1.)

        kernel_vals = np.array([lkernel[int(kbins*ll)] for ll in boverh[ok]])

        Z_los_SD[ii] = np.sum((thisgmass[ok]*thisgZ[ok]/(thisgsml[ok]*thisgsml[ok]))*kernel_vals) #in units of Msun/Mpc^2

    Z_los_SD*=conv #in units of Msun/pc^2

    return Z_los_SD


def get_recent_SFR(tag, t = 100, inp = 'FLARES'):

    #t is time in Myr
    #SFR in Msun/yr

    if inp == 'FLARES':
        sim_type = inp
        n = 40

    elif (inp == 'REF') or (inp == 'AGNdT9'):
        sim = F"./data/EAGLE_{inp}_sp_info.hdf5"
        sim_type = 'PERIODIC'
        n = 1

    else:
        ValueError(F"No input option of {inp}")


    for ii in range(n):
        if inp == 'FLARES':
            num = str(ii)
            if len(num) == 1:
                num =  '0'+num
            sim = F"./data/FLARES_{num}_sp_info.hdf5"

        with h5py.File(sim, 'r') as hf:

            S_len = np.array(hf[F'{tag}/Galaxy'].get('S_Length'), dtype = np.int64)
            S_mass = np.array(hf[F'{tag}/Particle'].get('S_Mass'), dtype = np.float64)
            S_age = np.array(hf[F'{tag}/Particle'].get('S_Age'), dtype = np.float64)*1e3 #Age is in Gyr,
                                                                         #so converting the array to Myr

        begin = np.zeros(len(S_len), dtype = np.int64)
        end = np.zeros(len(S_len), dtype = np.int64)
        begin[1:] = np.cumsum(S_len)[:-1]
        end = np.cumsum(S_len)

        SFR = np.zeros(len(begin))

        for jj, kk in enumerate(begin):

            this_age = S_age[begin[jj]:end[jj]]
            this_mass = S_mass[begin[jj]:end[jj]]
            ok = np.where(this_age <= t)[0]
            if len(ok) > 0:

                SFR[jj] = np.sum(this_mass[ok])/(t*1e6)

        fl = flares(sim, sim_type)
        fl.create_dataset(SFR, F"{tag}/Galaxy/SFR/SFR_{t}",
        desc = F"SFR of the galaxy averaged over the last {t}Myr", unit = "Msun/yr")

    print (F"Saved the SFR averaged over {t}Myr with tag {tag} for {inp} to file")


#Weighted quantiles
def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """
    Taken from From https://stackoverflow.com/a/29677616/1718096

    Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """

    # do some housekeeping
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    # if not sorted, sort values array
    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]


    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)



def binned_weighted_quantile(x,y,weights,bins,quantiles):

    # if ~isinstance(quantiles,list):
    #     quantiles = [quantiles]

    out = np.full((len(bins)-1,len(quantiles)),np.nan)
    for i,(b1,b2) in enumerate(zip(bins[:-1],bins[1:])):
        mask = (x >= b1) & (x < b2)
        if np.sum(mask) > 0:
            out[i,:] = weighted_quantile(y[mask],quantiles,sample_weight=weights[mask])

    return np.squeeze(out)
