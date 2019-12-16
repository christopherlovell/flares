import sys

ii, tag = sys.argv[1], sys.argv[2]
print (ii, tag)

num = str(ii)
tag = str(tag)

if len(num) == 1:
    num =  '0'+num

from mpi_save_to_hdf5 import save_to_hdf5

# save_to_hdf5(num, '010_z005p000', inp='GEAGLE')
# save_to_hdf5(num, '009_z006p000', inp='GEAGLE')
# save_to_hdf5(num, '008_z007p000', inp='GEAGLE')
# save_to_hdf5(num, '007_z008p000', inp='GEAGLE')
#save_to_hdf5(num, '006_z009p000', inp='GEAGLE')
save_to_hdf5(num, tag, inp='GEAGLE')
