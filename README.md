# First Light And Reionisation Epoch Simulations (FLARES)

A python convenience module for working with FLARES resimulation data.

## Requirements

- numpy
- h5py
- scipy 
- schwimmbad
- [eagle_IO](https://github.com/flaresimulations/eagle_IO)
- [SynthObs](https://github.com/stephenmwilkins/SynthObs) 

## Installation

Run the following to add this module *permanently* to your path:

    export PYTHONPATH=$PYTHONPATH:/path/to/this/directory/flares

You can then just run

    import flares

in any other scripts to get the flares class and associated functionality.

## Tutorial

`flares.py` contains the `flares` class, which contains a lot of useful functionality for anlysing the resims. The most important information is the specified halos (`flares.halos`) and snapshots (`flares.tags`) you wish to analyse; these should be updated as new resims are completed.

`download_methods.py` fetches the specified arrays from all resims and puts them in a single hdf5 file in the `data/` folder. Simply update these scripts with the data you wish to download, specify if you wish to `overwrite` any existing data. Run this ecript using the batchscript provided `download_particles.cosma.sh` for getting the particle data and run `download_subfind.py` just for the subfind data. Then run `create_UVgrid.cosma.sh` to get the value of kappa and use `download_phot.cosma.sh` to extract the photometry information. These are made into a single file `flares.hdf5` with the data structure as `Resim_num/Property_type/Property' where Resim_num is the number of the resim in the FLARES (see [here](https://docs.google.com/spreadsheets/d/1NzQee05rNCml1YEKXuD8L9JOW5Noh8oj9K9bcS2RQlY/edit?usp=sharing)), Property_type can be either Galaxy (like stellar mass, sfr, etc) or Particle (individual properties of gas/stellar particles) and Property is the required property. 



### Example

Once the data is downloaded, you can use it as so,

```
import flares
fl = flares.flares('./data/flares.hdf5', sim_type='FLARES')

mstar = fl.load_dataset('Mstar_30', arr_type='Galaxy')

halo = fl.halos[0]
tag = fl.tags[0]

print (mstar[halo][tag][:10])
```

Extracting stellar particle information,

```
import numpy as np
import h5py
fname = './data/flares.hdf5'
num = '00'
with h5py.File(fname, 'r') as hf:
    S_len = np.array(hf[num+'/'+tag+'/Galaxy'].get('S_Length'), dtype = np.int64)
    S_mass = np.array(hf[num+'/'+tag+'/Particle'].get('S_Mass'), dtype = np.float64)
    S_Z = np.array(hf[num+'/'+tag+'/Particle'].get('S_Z'), dtype = np.float64)
    S_age = np.array(hf[num+'/'+tag+'/Particle'].get('S_Age'), dtype = np.float64)*1e3

begin = np.zeros(len(S_len), dtype = np.int64)
end = np.zeros(len(S_len), dtype = np.int64)
begin[1:] = np.cumsum(S_len)[:-1]
end = np.cumsum(S_len)

#Age of all particles belonging to first galaxy in resim region 'num'

print (S_age[begin[0]:end[0]])
```
