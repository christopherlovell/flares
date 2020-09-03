
from download_methods import extract_subfind_info

properties = {'properties': ['MassType'], #'BlackHoleMass','BlackHoleMassAccretionRate'],
              'conv_factor': [1e10],
              'save_str': ['MassType']} #BlackHoleMass','BlackHoleMassAccretionRate']}


extract_subfind_info('data/flares_copy.hdf5', properties=properties, overwrite=True, verbose=True)

