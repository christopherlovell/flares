import os
import numpy as np
from scipy.spatial.distance import cdist

class flares:

    def __init__(self):

        self.halos = ['0000','0001','0002','0003','0005','0014','0015',
                 '0016','0017','0018','0019','0020','0021','0022',
                 '0023','0024','0025','0026','0027','0028','0029','0030']
        
        self.tags = ['000_z015p048','001_z012p762','002_z011p146', 
                     '003_z009p934','004_z008p985','005_z008p218', 
                     '006_z007p583','007_z007p047','008_z006p587',  
                     '009_z006p188','010_z005p837']

        self.alt_tags = ['000_z015p048','002_z012p762',
                         '003_z011p146','004_z009p934','005_z008p985', 
                         '006_z008p218','007_z007p583','008_z007p047', 
                         '009_z006p587','010_z006p188','011_z005p837'] # '001_z013p989',
        
        self.directory = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/'


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
 

