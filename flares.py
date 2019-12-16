import os
import numpy as np
from scipy.spatial.distance import cdist
import scipy.stats
import h5py

class flares:

    def __init__(self):

        self.halos = np.array([
                 '0000','0001','0002','0003','0004',
                 '0005','0006','0007','0008','0009',
                 '0010','0011','0012','0013','0014',
                 '0015','0016','0017','0018','0019',
                 '0020','0021','0022','0023','0024',
                 '0025','0026','0027','0028','0029',
                 '0030','0031','0032','0033','0034',
                 '0035','0036','0037','0038','0039'])
        
        self.tags = np.array(['000_z015p000','001_z014p000','002_z013p000',
                              '003_z012p000','004_z011p000','005_z010p000',
                              '006_z009p000','007_z008p000','008_z007p000',
                              '009_z006p000','010_z005p000','011_z004p770'])
        
        self.directory = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/'
        self.ref_directory = '/cosma5/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data'
        self.agn_directory = '/cosma5/data/Eagle/ScienceRuns/Planck1/L0050N0752/PE/S15_AGNdT9/data'

        self.ref_tags = np.array(['001_z015p132','002_z009p993','003_z008p988',
                                  '004_z008p075','005_z007p050','006_z005p971',
                                  '007_z005p487','008_z005p037','009_z004p485'])

        self.data = "%s/data"%(os.path.dirname(os.path.realpath(__file__)))

        ## update with weights file location
        self.weights = '/cosma7/data/dp004/dc-love2/codes/ic_selection/weights_grid.txt'


    def check_snap_exists(self,halo,snap):
        test_str = self.directory + 'geagle_' + halo + '/data/snapshot_' + snap 
        return os.path.isdir(test_str)
        

    def cutout(self, c0, c1, radius):
        
        if type(c0) is not np.ndarray:
            c0 = np.asarray(c0)
        
        if type(c1) is not np.ndarray:
            c1 = np.asarray(c1)

        if len(c0.shape) < 2:
            c0 = np.expand_dims(c0,axis=0)

        if len(c1.shape) < 2:
            c1 = np.expand_dims(c1,axis=0)

        return cdist(c0,c1).flatten() < radius

    
    def cut_radius(self, x, y, z, x0, y0, z0, threshold = 0.1, cut = 2):
        # Calculates the maximum cut out radius that leaves a 
        # sufficient gap to the edge of the high resolution region.
        # Tests that the region is close to spherical, otherwise a 
        # more sophisticated cut out must be constructed.
        #
        # Args:
        # - x, y, z: coordinate vectors
        # - x0, y0, z0: centre of simulation / high resolution region
        # - threshold: threshold ratio of extent in both directions, tested along each axis
        # - cut: how far in to cut out, cMpc (h less)
        #
        # Returns:
        # - minimum distance to edge minus cut boundary
        # - if threshold exceeded, returns False
    
        xmax = max(x) - x0
        ymax = max(y) - y0
        zmax = max(z) - z0
    
        xmin = x0 - min(x)
        ymin = y0 - min(y)
        zmin = z0 - min(z)
    
        if ((abs(xmax / xmin -1) > threshold) or (abs(ymax / ymin -1) > threshold) or (abs(zmax / zmin -1) > threshold)):
            print("Error! High resolution region highly non-spherical.")
            print('xmax: ', xmax, ', xmin: ', xmin)
            print('ymax: ', ymax, ', xmin: ', ymin)
            print('zmax: ', zmax, ', xmin: ', zmin)
            return False
    
        return(min(xmax,ymax,zmax,xmin,ymin,zmin) - cut)
 

    def calc_df(self, mstar, tag, volume, massBinLimits):

        hist, dummy = np.histogram(np.log10(mstar), bins = massBinLimits)
        hist = np.float64(hist)
        phi = (hist / volume) / (massBinLimits[1] - massBinLimits[0])
        # phi_sigma = poisson_confidence_intervals(hist,0.68) / volume) /\
        #             (massBinLimits[1] - massBinLimits[0])
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
            
            mask = (ps <= p)

            err_up = np.abs(np.log10(p) - np.log10(p + ps))
            err_lo = np.abs(np.log10(p) - np.log10(p - ps))

            return err_up, err_lo, mask

        
        err_up, err_lo, mask = yerr(phi,phi_sigma)

        err_lo[~mask] = 0.5
        err_up[~mask] = 0.5
        # # set phi to upper limit where no lower limit
        # # phi[~mask] = 10**(np.log10(phi[~mask]) + err_up[~mask])

        ax.errorbar(np.log10(massBins[phi > 0.]),
                np.log10(phi[phi > 0.]),
                yerr=[err_up[phi > 0.],
                      err_lo[phi > 0.]],
                uplims=(~mask[phi > 0.]),
                label=label, c=color, alpha=alpha, **kwargs)


    @staticmethod
    def poisson_confidence_interval(n,p):
        """
        http://ms.mcmaster.ca/peter/s743/poissonalpha.html
            
        e.g. p=0.68 for 1 sigma
        
        agrees with http://hyperphysics.phy-astr.gsu.edu/hbase/math/poifcn.html
          
        see comments on JavaStat page
        
         scipy.stats.chi2.ppf((1.-p)/2.,2*n)/2. also known
        """
        
        if ~hasattr(n, "__iter__"): # check it's an array
            n = np.array(n)

        interval = np.zeros((len(n),2))
        mask = n > 0.

        interval[mask,0]=scipy.stats.chi2.ppf((1.-p)/2.,2*n[mask])/2.
        interval[mask,1]=scipy.stats.chi2.ppf(p+(1.-p)/2.,2*(n[mask]+1))/2.
            
        # work out the case for n=0
        ul=(1.-p)/2.
        
        prev=1.0
        for a in np.arange(0.,5.0,0.001):
        
            cdf=scipy.stats.poisson.cdf(0.,a)
        
            if cdf<ul and prev>ul:
                i=a
        
            prev=cdf
        
        interval[~mask]=[0.,i]
        
        return np.array(interval)



    """
    Utilities for loading and saving nested dictionaries recursively

    see https://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py
    """

    def save_dict_to_hdf5(self, dic, filename, groupname='default', overwrite=False):
        """
        ....
        """
        if groupname=='default': print("Saving to `default` group")
        with h5py.File(filename, 'a') as h5file:
            self.recursively_save_dict_contents_to_group(h5file, groupname+'/', dic, overwrite=overwrite)
    
    def recursively_save_dict_contents_to_group(self, h5file, path, dic, overwrite=False):
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
                self.recursively_save_dict_contents_to_group(h5file, path + key + '/', item, overwrite=overwrite)
            else:
                raise ValueError('Cannot save %s type'%type(item))
    
    def load_dict_from_hdf5(self, filename, group=''):
        """
        ....
        """
        with h5py.File(filename, 'r') as h5file:
            return self.recursively_load_dict_contents_from_group(h5file, group+'/')
    
    
    def recursively_load_dict_contents_from_group(self, h5file, path):
        """
        ....
        """
        ans = {}
        for key, item in h5file[path].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                ans[key] = item[()] # .value
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = self.recursively_load_dict_contents_from_group(h5file, path + key + '/')
        return ans


