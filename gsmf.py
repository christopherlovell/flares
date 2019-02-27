
import eagle as E
import pickle as pcl
import numpy as np


directory = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_0000/data'

tags = ['003_z009p934','004_z008p985','005_z008p218']


mstar_30 = E.readArray('SUBFIND', directory, tag, '/Subhalo/ApertureMeasurements/Mass/030kpc')[:,4] * 1e10
# mstar = E.readArray('SUBFIND', directory, tag, '/Subhalo/Stars/Mass', numThreads=1) * 1e10
# sf_metal = E.readArray('SUBFIND', directory, tag, '/Subhalo/SF/Metallicity', numThreads=1)
# sf_mass = E.readArray('SUBFIND', directory, tag, '/Subhalo/SF/Mass', numThreads=1) * 1e10
# sfr_30 = E.readArray('SUBFIND', directory, tag, '/Subhalo/ApertureMeasurements/SFR/030kpc')
sfr = E.readArray('SUBFIND', directory, tag, '/Subhalo/StarFormationRate', numThreads=1)
# cop = E.readArray('SUBFIND', directory, tag, '/Subhalo/CentreOfPotential', numThreads=1, physicalUnits=True)
# hmr = E.readArray('SUBFIND', directory, tag, '/Subhalo/HalfMassRad', numThreads=1)[:,0]







