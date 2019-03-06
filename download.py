import pickle as pcl
import sys

import numpy as np
import eagle as E

from misc import *


sfr = {halo: {} for halo in halos}
mstar = {halo: {} for halo in halos}
coods = {halo: {} for halo in halos}
# centrals = {halo: {} for halo in mpcdf.halos}
# bhmass = {halo: {} for halo in mpcdf.halos}
# bhaccr = {halo: {} for halo in mpcdf.halos}
# tmass = {halo: {} for halo in mpcdf.halos}
# Zstars = {halo: {} for halo in mpcdf.halos}
# ZgasSF = {halo: {} for halo in mpcdf.halos}
# ZgasNSF = {halo: {} for halo in mpcdf.halos}
# m200 = {halo: {} for halo in mpcdf.halos}


for halo in halos:
    for tag in tags:

        print(halo, tag)
        halodir = directory+'/geagle_'+halo+'/data/'

        mstar[halo][tag] = E.readArray("SUBFIND", halodir, tag, "/Subhalo/Stars/Mass", numThreads=1, noH=True)
        sfr[halo][tag] = E.readArray("SUBFIND", halodir, tag, "/Subhalo/StarFormationRate", numThreads=1, noH=True)
        coods[halo][tag] = E.readArray("SUBFIND", halodir, tag, "/Subhalo/CentreOfPotential", numThreads=1, noH=True, physicalUnits=False)



        # bhmass[halo][tag] = E.readArray("SUBFIND", halodir, tag, "/Subhalo/BlackHoleMass", numThreads=1, noH=True)
        # tmass[halo][tag] = E.readArray("SUBFIND", halodir, tag, "/Subhalo/Mass", numThreads=1, noH=True)
        # Zstars[halo][tag] = E.readArray("SUBFIND", halodir, tag, "/Subhalo/Stars/Metallicity", numThreads=1, noH=True)
        # ZgasSF[halo][tag] = E.readArray("SUBFIND", halodir, tag, "/Subhalo/SF/Metallicity", numThreads=1, noH=True)
        # grpn = E.readArray("SUBFIND", halodir, tag, "/Subhalo/GroupNumber", numThreads=1)
        # m200[halo][tag] = E.readArray("SUBFIND_GROUP", halodir, tag, "/FOF/Group_M_Mean200", numThreads=1, noH=True)[grpn - 1]

    #     bhaccr[halo][tag] = E.readArray("SUBFIND", halodir, tag, "/Subhalo/BlackHoleMassAccretionRate", numThreads=1, noH=True)

    #     centrals[halo][tag] = (E.readArray("SUBFIND", halodir, tag, "/Subhalo/SubGroupNumber", numThreads=1, noH=True) == 0)
    #     ZgasNSF[halo][tag] = E.readArray("SUBFIND", halodir, tag, "/Subhalo/NSF/Metallicity", numThreads=1, noH=True)


pcl.dump(mstar, open('pcls/mstar.p','wb'))
pcl.dump(sfr, open('pcls/sfr.p','wb'))
pcl.dump(coods, open('pcls/coods.p','wb'))
