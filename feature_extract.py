import numpy as np
import h5py
import config
import high_trans
import second_trans

full_size = config.TEST_SIZE

if full_size > 10000:
    print("Using batch size 10000")
    config.TEST_SIZE = 10000

print(config.TRANSFERSET_PATH)
hf = h5py.File(config.TRANSFERSET_PATH, 'r+')
x = hf['X']
y = hf['Y']

hf_out = h5py.File(config.TRANSFERSET_FEATURESET_PATH, 'w')

test_out = hf_out.create_dataset('X', shape=config.SHAPE, dtype=np.complex64, chunks=True)
label_out = hf_out.create_dataset('Y', shape=(config.SHAPE[0], 24), dtype=np.int64, chunks=True)
# RadioML only
#z = hf['Z']
#snr_out = hf_out.create_dataset('Z', shape=(config.SHAPE[0], 1), dtype=np.int64, chunks=True)

for start in range(0, full_size, config.TEST_SIZE):
    end = min(start + config.TEST_SIZE, full_size)
    config.TEST_SIZE = end - start

    print(f"Processing batch {start}-{end}")
   
    batch = x[start:end]
    batch_labels = y[start:end]
    #batch_snr = z[start:end]

    # Get signal from IQ
    s = batch[:,:,0] + 1j*batch[:,:,1]

    # Get TEST_SIZE samples from s
    data_test = np.asarray(s[:, :])

    if config.FEATURE_TO_EXTRACT in ["element_HOS", "RD_CTCF"]:
        test = high_trans.testrun(config.FEATURE_TO_EXTRACT, data_test)
    elif config.FEATURE_TO_EXTRACT in ["SCD", "CHTC", "CCSD"]:
        test = second_trans.testrun(config.FEATURE_TO_EXTRACT, data_test, config.EXTRACT_PROFILE)
    else:
        print("Feature not supported :(")
        exit

    test_out[start:end] = np.asarray(test, dtype=np.complex64)
    label_out[start:end] = batch_labels
    snr_out[start:end] = batch_snr

hf.close()
hf_out.close()
print(f"Saved to {config.TRANSFERSET_FEATURESET_PATH}")
