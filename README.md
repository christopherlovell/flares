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

`download_methods.py` fetches the specified arrays from all resims and puts them in a single hdf5 file in the `data/` folder. Simple update these scripts with the data you wish to download, specify if you wish to `overwrite` any existing data. Run this ecript using the batchscript provided `download_particles.cosma.sh` for getting the particle data and run `download_subfind.py` just for the subfind data. Then run `create_UVgrid.cosma.sh` to get the value of kappa and use `download_phot.cosma.sh` to extract the photometry information.



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
