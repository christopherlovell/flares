import numpy as np
import math
from astropy import units as u
import timeit
import gc
from mpi4py import MPI

import sys
import eagle_IO.eagle_IO as E
import flares

norm = np.linalg.norm
conv = (u.solMass/u.Mpc**2).to(u.solMass/u.pc**2)


def rotation_matrix(axis, theta):
    """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        `https://stackoverflow.com/questions/6802577/rotation-of-3d-vector`
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def make_faceon(cop, this_g_cood, this_g_mass, this_g_vel):
####Incomplete####

    this_g_cood -= cop
    ok = np.where(np.sqrt(np.sum(this_g_cood**2, axis = 1)) <= 0.03)
    this_g_cood = this_g_cood[ok]
    this_g_mass = this_g_mass[ok]
    this_g_vel = this_g_vel[ok]

    #Get the angular momentum unit vector
    L_tot = np.array([this_g_mass]).T*np.cross(this_g_cood, this_g_vel)
    L_tot_mag = np.sqrt(np.sum(np.nansum(L_tot, axis = 0)**2))
    L_unit = np.sum(L_tot, axis = 0)/L_tot_mag

    #z-axis as the face-on axis
    theta = np.arccos(L_unit[2])

    vec = np.cross(np.array([0., 0., 1.]), L_unit)
    r = rotation_matrix(vec, theta)

    #this_sp_cood *= 1e3
    new = np.array(this_s_cood.dot(r))

    return new



def extract_subfind_info(fname='data/flares.hdf5', inp='FLARES', 
                         properties = {'properties': ['MassType'], 
                                       'conv_factor': [1e10],
                                       'save_str': ['MassType']},
                         overwrite=False, threads=8, verbose=False):

    fl = flares.flares(fname,inp)
    indices = fl.load_dataset('Indices')

    for halo in fl.halos:

        print(halo)

        fl.create_group(halo)

        for tag in fl.tags:

            fl.create_group('%s/%s'%(halo,tag))
            fl.create_group('%s/%s/Galaxy'%(halo,tag))

            print(tag)

            halodir = fl.directory+'/GEAGLE_'+halo+'/data/'

            props = properties['properties']
            conv_factor = properties['conv_factor']
            save_str = properties['save_str']

            for _prop,_conv,_save in zip(props,conv_factor,save_str):
            
                if (fl._check_hdf5('%s/%s/Subhalo/%s'%(halo,tag,_prop)) is False) |\
                         (overwrite == True):

                    _arr = E.read_array("SUBFIND", halodir, tag,
                                        "/Subhalo/%s"%_prop,
                                        numThreads=threads, noH=True) * _conv
   

                    fl.create_dataset(_arr[indices[halo][tag].astype(int)], _save,
                                      '%s/%s/Galaxy/'%(halo,tag), overwrite=True, verbose=verbose)



def extract_info(num, tag, kernel='sph-anarchy', inp='FLARES'):
    """
    Args:
        num (str): the FLARES/G-EAGLE id of the sim; eg: '00', '01', ...
        tag (str): the file tag; eg: '000_z015p00', '001_z014p000',...., '011_z004p770'
    Selects only galaxies with more than 100 star+gas particles inside 30pkpc
    """

    ## MPI parameters
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print (F"Extracing information from {inp} {num} {tag}")

    if inp == 'FLARES':
        sim_type = 'FLARES'
        fl = flares.flares(fname = './data2/',sim_type=sim_type)
        num = str(num)
        if len(num) == 1:
            num =  '0'+num
        dir = fl.directory
        sim = F"{dir}GEAGLE_{num}/data/"

    elif inp == 'REF':
        sim_type = 'PERIODIC'
        fl = flares.flares(fname = './data/',sim_type=sim_type)
        sim = fl.ref_directory

    elif inp == 'AGNdT9':
        sim_type = 'PERIODIC'
        fl = flares.flares(fname = './data/',sim_type=sim_type)
        sim = fl.agn_directory

    else:
        ValueError("Type of input simulation not recognized")

    if rank == 0:
        print (F"Sim location: {sim}, tag: {tag}")

    #Simulation properties
    z = E.read_header('SUBFIND', sim, tag, 'Redshift')
    a = E.read_header('SUBFIND', sim, tag, 'ExpansionFactor')

    ## Galaxy global properties
    M200 = E.read_array('SUBFIND', sim, tag, 'FOF/Group_M_Crit200', numThreads=4, noH=True, physicalUnits=True)*1e10
    M500 = E.read_array('SUBFIND', sim, tag, '/FOF/Group_M_Crit500', numThreads=4, noH=True, physicalUnits=True)*1e10
    M2500 = E.read_array('SUBFIND', sim, tag, '/FOF/Group_M_Crit2500', numThreads=4, noH=True, physicalUnits=True)*1e10
    SubhaloMass = E.read_array('SUBFIND', sim, tag, '/Subhalo/Mass', numThreads=4, noH=True, physicalUnits=True)*1e10
    mstar = E.read_array('SUBFIND', sim, tag, '/Subhalo/ApertureMeasurements/Mass/030kpc', numThreads=4, noH=True, physicalUnits=True)[:,4]*1e10
    sgrpno = E.read_array('SUBFIND', sim, tag, '/Subhalo/SubGroupNumber', numThreads=4)
    grpno = E.read_array('SUBFIND', sim, tag, '/Subhalo/GroupNumber', numThreads=4)
    vel = E.read_array('SUBFIND', sim, tag, '/Subhalo/Velocity', noH=True, physicalUnits=True, numThreads=4)

    if inp == 'FLARES':
        ## Selecting the subhalos within our region
        cop = E.read_array('SUBFIND', sim, tag, '/Subhalo/CentreOfPotential', noH=False, physicalUnits=False, numThreads=4) #units of cMpc/h
        cen, r, min_dist = fl.spherical_region(sim, tag)  #units of cMpc/h
        indices = np.where(np.logical_and(mstar >= 10**7., np.sqrt(np.sum((cop-cen)**2, axis = 1))<=fl.radius) == True)[0]

    else:
        indices = np.where(mstar >= 10**7.)[0]

    cop = E.read_array('SUBFIND', sim, tag, '/Subhalo/CentreOfPotential', noH=True, physicalUnits=True, numThreads=4)
    sfr_inst =  E.read_array('SUBFIND', sim, tag, '/Subhalo/ApertureMeasurements/SFR/030kpc', numThreads=4, noH=True, physicalUnits=True)

    ## Particle properties
    sp_cood = E.read_array('PARTDATA', sim, tag, '/PartType4/Coordinates', noH=True, physicalUnits=True, numThreads=4)
    gp_cood = E.read_array('PARTDATA', sim, tag, '/PartType0/Coordinates', noH=True, physicalUnits=True, numThreads=4)
    sp_sgrpn = E.read_array('PARTDATA', sim, tag, '/PartType4/SubGroupNumber', numThreads=4)
    sp_grpn = E.read_array('PARTDATA', sim, tag, '/PartType4/GroupNumber', numThreads=4)


    sp_mass = E.read_array('PARTDATA', sim, tag, '/PartType4/Mass', noH=True, physicalUnits=True, numThreads=4) * 1e10
    sp_mass_init = E.read_array('PARTDATA', sim, tag, '/PartType4/InitialMass', noH=True, physicalUnits=True, numThreads=4) * 1e10
    sp_smooth_Z = E.read_array('PARTDATA', sim, tag, '/PartType4/SmoothedMetallicity', numThreads=4)
    sp_Z = E.read_array('PARTDATA', sim, tag, '/PartType4/Metallicity', numThreads=4)
    sp_vel = E.read_array('PARTDATA', sim, tag, '/PartType4/Velocity', noH=True, physicalUnits=True, numThreads=4)
    sp_sl = E.read_array('PARTDATA', sim, tag, '/PartType4/SmoothingLength', noH=True, physicalUnits=True, numThreads=4)
    sp_ft = E.read_array('PARTDATA', sim, tag, '/PartType4/StellarFormationTime', noH=True, physicalUnits=True, numThreads=4)
    sp_ids = E.read_array('PARTDATA', sim, tag, '/PartType4/ParticleIDs', noH=True, physicalUnits=True, numThreads=4)


    gp_sgrpn = E.read_array('PARTDATA', sim, tag, '/PartType0/SubGroupNumber', numThreads=4)
    gp_grpn = E.read_array('PARTDATA', sim, tag, '/PartType0/GroupNumber', numThreads=4)
    gp_mass = E.read_array('PARTDATA', sim, tag, '/PartType0/Mass', noH=True, physicalUnits=True, numThreads=4) * 1e10
    gp_smooth_Z = E.read_array('PARTDATA', sim, tag, '/PartType0/SmoothedMetallicity', numThreads=4)
    gp_Z = E.read_array('PARTDATA', sim, tag, '/PartType0/Metallicity', numThreads=4)
    gp_vel = E.read_array('PARTDATA', sim, tag, '/PartType0/Velocity', noH=True, physicalUnits=True, numThreads=4)
    gp_sl = E.read_array('PARTDATA', sim, tag, '/PartType0/SmoothingLength', noH=True, physicalUnits=True, numThreads=4)
    gp_ids = E.read_array('PARTDATA', sim, tag, '/PartType0/ParticleIDs', noH=True, physicalUnits=True, numThreads=4)

    kinp = np.load('./data/kernel_{}.npz'.format(kernel), allow_pickle=True)
    lkernel = kinp['kernel']
    header = kinp['header']
    kbins = header.item()['bins']

    if rank == 0:
        print('\n\t\t\t###################################\nKernel used is `{}` and the number of bins in the look up table is {}\n\t\t\t###################################\n'.format(header.item()['kernel'], kbins))


    gc.collect()

    comm.Barrier()

    if rank == 0:
        if inp != 'FLARES': num = ''
        print("Extracting required properties for {} subhalos from {} at z = {}".format(len(indices), inp+num, z))

    part = int(len(indices)/size)

    if rank!=size-1:
        thisok = indices[rank*part:(rank+1)*part]
    else:
        thisok = indices[rank*part:]

    tsnum = np.zeros(len(thisok)+1).astype(int)
    tgnum = np.zeros(len(thisok)+1).astype(int)
    ind = np.array([])

    sn = len(sp_mass)
    gn = len(gp_mass)

    tsmass = np.empty(sn)
    tsmass_init = np.empty(sn)
    tgmass = np.empty(gn)

    tsid = np.empty(sn)
    tgid = np.empty(gn)

    tsvel = np.empty((sn,3))
    tgvel = np.empty((gn,3))

    tsZ = np.empty(sn)
    tsZ_smooth = np.empty(sn)
    tgZ = np.empty(gn)
    tgZ_smooth = np.empty(gn)

    tscood = np.empty((sn,3))
    tgcood = np.empty((gn,3))

    ts_sml = np.empty(sn)
    tg_sml = np.empty(gn)

    tsage = np.empty(sn)
    tZ_los = np.empty(sn)

    gc.collect()

    #Getting the indices (major) and the calculation of Z-LOS are the bottlenecks of this code. Maybe try
    #cythonizing the Z-LOS part. Don't know how to change the logical operation part.
    kk = 0

    for ii, jj in enumerate(thisok):

        start = timeit.default_timer()

        s_ok = np.where((sp_sgrpn-sgrpno[jj]==0) & (sp_grpn-grpno[jj]==0))[0]
        s_ok = s_ok[norm(sp_cood[s_ok]-cop[jj],axis=1)<=0.03]
        g_ok = np.where((gp_sgrpn-sgrpno[jj]==0) & (gp_grpn-grpno[jj]==0))[0]
        g_ok = g_ok[norm(gp_cood[g_ok]-cop[jj],axis=1)<=0.03]

        stop = timeit.default_timer()

        if len(s_ok) + len(g_ok) >= 100:

            print ("Calculating indices took {}s".format(np.round(stop - start,6)))
            #Use the `make_faceon` function here to transform to face-on coordinates. Need to add more
            #to make use of it. At the moment everything along the z-axis

            start = timeit.default_timer()
            Z_los_SD = flares.get_Z_LOS(sp_cood[s_ok], gp_cood[g_ok], gp_mass[g_ok], gp_smooth_Z[g_ok], gp_sl[g_ok], lkernel, kbins)
            stop = timeit.default_timer()
            print ("Calculating Z_los took {}s".format(np.round(stop - start,6)))

            start = timeit.default_timer()
            tsnum[kk+1] = len(s_ok)
            tgnum[kk+1] = len(g_ok)

            scum = np.cumsum(tsnum)
            gcum = np.cumsum(tgnum)
            sbeg = scum[kk]
            send = scum[kk+1]
            gbeg = gcum[kk]
            gend = gcum[kk+1]

            tscood[sbeg:send] = sp_cood[s_ok]
            tgcood[gbeg:gend] = gp_cood[g_ok]

            tsid[sbeg:send] = sp_ids[s_ok]
            tgid[gbeg:gend] = gp_ids[g_ok]

            tsmass[sbeg:send] = sp_mass[s_ok]
            tsmass_init[sbeg:send] = sp_mass_init[s_ok]
            tgmass[gbeg:gend] = gp_mass[g_ok]

            tsvel[sbeg:send] = sp_vel[s_ok]
            tgvel[gbeg:gend] = gp_vel[g_ok]

            tsZ[sbeg:send] = sp_Z[s_ok]
            tsZ_smooth[sbeg:send] = sp_smooth_Z[s_ok]
            tgZ[gbeg:gend] = gp_Z[g_ok]
            tgZ_smooth[gbeg:gend] = gp_smooth_Z[g_ok]

            ts_sml[sbeg:send] = sp_sl[s_ok]
            tg_sml[gbeg:gend] = gp_sl[g_ok]

            tsage[sbeg:send] = sp_ft[s_ok]
            tZ_los[sbeg:send] = Z_los_SD
            stop = timeit.default_timer()
            print ("Assigning arrays took {}s".format(np.round(stop - start,6)))
            gc.collect()

            kk+=1
        else:

            ind = np.append(ind, ii)

    ##End of loop ii, jj##
    del sp_sgrpn, sp_grpn, sp_mass, sp_mass_init, sp_Z, sp_smooth_Z, sp_cood, sp_sl, sp_ft, gp_sgrpn, gp_grpn, gp_mass, gp_Z, gp_smooth_Z, gp_cood, gp_sl

    gc.collect()


    thisok = np.delete(thisok, ind.astype(int))

    tstot = np.sum(tsnum)
    tgtot = np.sum(tgnum)

    tsnum = tsnum[1:len(thisok)+1]
    tgnum = tgnum[1:len(thisok)+1]

    tscood = tscood[:tstot]
    tgcood = tgcood[:tgtot]

    tsid = tsid[:tstot]
    tgid = tgid[:tgtot]

    tsmass = tsmass[:tstot]
    tsmass_init = tsmass_init[:tstot]
    tgmass = tgmass[:tgtot]

    tsvel = tsvel[:tstot]
    tgvel = tgvel[:tgtot]

    tsZ = tsZ[:tstot]
    tsZ_smooth = tsZ_smooth[:tstot]
    tgZ = tgZ[:tgtot]
    tgZ_smooth = tgZ_smooth[:tgtot]

    ts_sml = ts_sml[:tstot]
    tg_sml = tg_sml[:tgtot]

    tsage = fl.get_age(tsage[:tstot], z, 4)
    tZ_los = tZ_los[:tstot]

    comm.Barrier()

    if rank == 0:

        print ("Gathering data from different processes")

    indices = comm.gather(thisok, root=0)

    snum = comm.gather(tsnum, root=0)
    gnum = comm.gather(tgnum, root=0)
    scood = comm.gather(tscood, root=0)
    gcood = comm.gather(tgcood, root=0)
    sid = comm.gather(tsid, root=0)
    gid = comm.gather(tgid, root=0)
    smass = comm.gather(tsmass, root=0)
    smass_init = comm.gather(tsmass_init, root=0)
    gmass = comm.gather(tgmass, root=0)
    svel = comm.gather(tsvel, root=0)
    gvel = comm.gather(tgvel, root=0)
    sZ = comm.gather(tsZ, root=0)
    sZ_smooth = comm.gather(tsZ_smooth, root=0)
    gZ = comm.gather(tgZ, root=0)
    gZ_smooth = comm.gather(tgZ_smooth, root=0)
    s_sml = comm.gather(ts_sml, root=0)
    g_sml = comm.gather(tg_sml, root=0)
    sage = comm.gather(tsage, root=0)
    Z_los = comm.gather(tZ_los, root=0)


    if rank == 0:

        print ("Gathering completed")
        indices = np.concatenate(np.array(indices))

        snum = np.concatenate(np.array(snum))
        gnum = np.concatenate(np.array(gnum))

        scood = np.concatenate(np.array(scood), axis = 0)/a
        gcood = np.concatenate(np.array(gcood), axis = 0)/a

        sid = np.concatenate(np.array(sid))
        gid = np.concatenate(np.array(gid))

        smass = np.concatenate(np.array(smass))
        smass_init = np.concatenate(np.array(smass_init))
        gmass = np.concatenate(np.array(gmass))

        svel = np.concatenate(np.array(svel), axis = 0)
        gvel = np.concatenate(np.array(gvel), axis = 0)

        sZ = np.concatenate(np.array(sZ))
        sZ_smooth = np.concatenate(np.array(sZ_smooth))
        gZ = np.concatenate(np.array(gZ))
        gZ_smooth = np.concatenate(np.array(gZ_smooth))

        s_sml = np.concatenate(np.array(s_sml))
        g_sml = np.concatenate(np.array(g_sml))

        sage = np.concatenate(np.array(sage))
        Z_los = np.concatenate(np.array(Z_los))

        ok_centrals = grpno[indices] - 1


    return indices, M200[ok_centrals], M500[ok_centrals], M2500[ok_centrals], SubhaloMass[indices], mstar[indices], cop[indices]/a, vel[indices], sfr_inst[indices], grpno[indices], sgrpno[indices], snum, gnum, scood, gcood, sid, gid, smass, smass_init, gmass, svel, gvel, sZ, sZ_smooth, gZ, gZ_smooth, s_sml, g_sml, sage, Z_los

##End of function `extract_info`

def save_to_hdf5(num, tag, kernel='sph-anarchy', inp='FLARES'):

    num = str(num)
    if inp == 'FLARES':
        if len(num) == 1:
            num =  '0'+num
        filename = 'data2/FLARES_{}_sp_info.hdf5'.format(num)
        sim_type = 'FLARES'

    elif inp == 'REF' or inp == 'AGNdT9':
        filename = F"data/EAGLE_{inp}_sp_info.hdf5"
        sim_type = 'PERIODIC'

    else:
        ValueError("Type of input simulation not recognized")

    ## MPI parameters
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:

        print ("#################    Saving required properties to hdf5     #############")
        print ("#################   Number of processors being used is {}   #############".format(size))


    indices, M200, M500, M2500, SubhaloMass, mstar, cop, vel, sfr_inst,  grpno, sgrpno, snum, gnum, scood, gcood, sid, gid, smass, smass_init, gmass, svel, gvel, sZ, sZ_smooth, gZ, gZ_smooth, s_sml, g_sml, sage, Z_los = extract_info(num, tag, kernel, inp)


    if rank == 0:

        print("Wrting out required properties to hdf5")

        fl = flares.flares(fname = filename,sim_type = sim_type)
        fl.create_group(tag)

        fl.create_group('{}/Galaxy'.format(tag))
        fl.create_group('{}/Particle'.format(tag))


        fl.create_dataset(indices, 'Indices', '{}/Galaxy'.format(tag), dtype = np.int64,
            desc = 'Index of the galaxy in the resimulation')

        fl.create_dataset(M200, 'M200', '{}/Galaxy'.format(tag),
            desc = 'M200 of the group', unit = 'Msun')
        fl.create_dataset(M500, 'M500', '{}/Galaxy'.format(tag),
            desc = 'M500 of the group', unit = 'Msun')
        fl.create_dataset(M2500, 'M2500', '{}/Galaxy'.format(tag),
            desc = 'M2500 of the group', unit = 'Msun')

        fl.create_dataset(SubhaloMass, 'SubhaloMass', '{}/Galaxy'.format(tag),
            desc = 'Subhalo mass of the sub-group', unit = 'Msun')
        fl.create_dataset(mstar, 'Mstar_30', '{}/Galaxy'.format(tag),
            desc = 'Stellar mass of the galaxy measured inside 30pkc aperture', unit = 'Msun')
        fl.create_dataset(cop.T, 'COP', '{}/Galaxy'.format(tag),
            desc = 'Centre Of Potential of the galaxy', unit = 'cMpc')
        fl.create_dataset(vel.T, 'Velocity', '{}/Galaxy'.format(tag),
            desc = 'Velocity of the galaxy', unit = 'km/s')
        fl.create_dataset(sfr_inst, 'SFR_inst_30', '{}/Galaxy'.format(tag),
            desc = 'Instantaneous sfr of the galaxy measured inside 30pkc aperture', unit = 'Msun/yr')
        fl.create_dataset(grpno, 'GroupNumber', '{}/Galaxy'.format(tag), dtype = np.int64,
            desc = 'Group Number of the galaxy')
        fl.create_dataset(sgrpno, 'SubGroupNumber', '{}/Galaxy'.format(tag), dtype = np.int64,
            desc = 'Subgroup Number of the galaxy')



        fl.create_dataset(snum, 'S_Length', '{}/Galaxy'.format(tag), dtype = np.int64,
            desc = 'Number of star particles inside 30pkpc')
        fl.create_dataset(gnum, 'G_Length', '{}/Galaxy'.format(tag), dtype = np.int64,
            desc = 'Number of gas particles inside 30pkpc')

        fl.create_dataset(scood.T, 'S_Coordinates', '{}/Particle'.format(tag),
            desc = 'Star particle coordinates', unit = 'cMpc')
        fl.create_dataset(gcood.T, 'G_Coordinates', '{}/Particle'.format(tag),
            desc = 'Gas particle coordinates', unit = 'cMpc')

        fl.create_dataset(sid, 'S_ID', '{}/Particle'.format(tag),
            desc = 'Star particle ID', unit = 'None')
        fl.create_dataset(gid, 'G_ID', '{}/Particle'.format(tag),
            desc = 'Gas particle ID', unit = 'None')

        fl.create_dataset(smass, 'S_Mass', '{}/Particle'.format(tag),
            desc = 'Star particle masses', unit = 'Msun')
        fl.create_dataset(smass_init, 'S_MassInitial', '{}/Particle'.format(tag),
            desc = 'Star particle masses at formation time', unit = 'Msun')
        fl.create_dataset(gmass, 'G_Mass', '{}/Particle'.format(tag),
            desc = 'Gas particle masses', unit = 'Msun')

        fl.create_dataset(svel, 'S_Vel', '{}/Particle'.format(tag),
            desc = 'Star particle peculiar velocity', unit = 'km/s')
        fl.create_dataset(gvel, 'G_Vel', '{}/Particle'.format(tag),
            desc = 'Gas particle peculiar velocity', unit = 'km/s')

        fl.create_dataset(sZ, 'S_Z', '{}/Particle'.format(tag),
            desc = 'Star particle metallicity', unit = 'No units')
        fl.create_dataset(sZ_smooth, 'S_Z_smooth', '{}/Particle'.format(tag),
            desc = 'Star particle smoothed metallicity', unit = 'No units')
        fl.create_dataset(gZ, 'G_Z', '{}/Particle'.format(tag),
            desc = 'Gas particle metallicity', unit = 'No units')
        fl.create_dataset(gZ_smooth, 'G_Z_smooth', '{}/Particle'.format(tag),
            desc = 'Gas particle smoothed metallicity', unit = 'No units')

        fl.create_dataset(s_sml, 'S_sml', '{}/Particle'.format(tag),
            desc = 'Star particle smoothing length', unit = 'pMpc')
        fl.create_dataset(g_sml, 'G_sml', '{}/Particle'.format(tag),
            desc = 'Gas particle smoothing length', unit = 'pMpc')

        fl.create_dataset(sage, 'S_Age', '{}/Particle'.format(tag),
            desc = 'Star particle age', unit = 'Gyr')
        fl.create_dataset(Z_los, 'S_los', '{}/Particle'.format(tag),
            desc = 'Star particle line-of-sight metal column density along the z-axis', unit = 'Msun/pc^2')


if __name__ == "__main__":

    ii, tag, inp = sys.argv[1], sys.argv[2], sys.argv[3]

    num = str(ii)
    tag = str(tag)
    inp = str(inp)

    if len(num) == 1:
        num =  '0'+num


    save_to_hdf5(num, tag, inp=inp)
