# First Light And REionisation Simulations (FLARES)

A python convenience module for working with FLARES resimulation data.

## Requirements

- numpy
- h5py
- scipy (optional, for cutout)

## Installation

Run the following to add this module *permanently* to your path:

    export PYTHONPATH=$PYTHONPATH:/path/to/this/directory/flares

You can then just run

    import flares

in any other scripts to get the flares class and associated functionality.

## Tutorial

`flares.py` contains the `flares` class, which contains a lot of useful functionality for anlysing the resims. The most important information is the specified halos (`flares.halos`) and snapshots (`flares.tags`) you wish to analyse; these should be updated as new resims are completed.

`download.py` fetches the specified arrays from all resims and puts them in a single hdf5 file in the `data/` folder. Similarly, `download_periodic.py` does this for any specified periodic volumes (Ref_100 and AGNdT9_50 by default). Simple update these scripts with the data you wish to download, specify if you wish to `overwrite` any existing data, and run directly:

```
python download.py
python download_periodic.py
```

### Example

Once the data is downloaded, you can use it as so,

```
import flares
fl = flares.flares()

fname = fl.data+'/flares.h5'
mstar = fl.load_dict_from_hdf5(fname, 'mstar')

halo = fl.halos[0]
tag = fl.tags[0]

print (mstar[halo][tag][:10])
```
