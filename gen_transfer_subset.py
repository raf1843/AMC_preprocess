import numpy as np
import h5py
import config

print(config.TRANSFERSET_PATH)
hf = h5py.File(config.TRANSFERSET_PATH, 'r+')
x = hf['X']
# Get signal from IQ
s = x[:,:,0] + 1j*x[:,:,1]
# Get 30 samples from s to match test size
data_test = np.asarray(s[:config.TEST_SIZE, :])
hf.close()

# write to hdf5 file
hf_out = h5py.File(config.TRANSFERSET_SUBSET_PATH, 'w')
hf_out.create_dataset('test', data=data_test)
hf_out.close()
