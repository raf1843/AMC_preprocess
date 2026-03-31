''' Datasets to test generalization '''
#TRANSFERSET_NAME = "COMS-1"
#TRANSFERSET_PATH = "../../data/vksdr/COMS-1_LRIT_multiple-images.hdf5"
#TRANSFERSET_SUBSET_PATH = "coms-1_subset.h5"
#TRANSFERSET_LABEL = [3]

#TRANSFERSET_NAME = "SATIQ"
#TRANSFERSET_PATH = "../../data/satiq/satiq_testdata_000.hdf5"
#TRANSFERSET_SUBSET_PATH = "satiq_subset.h5"
#TRANSFERSET_LABEL = [4]

#TRANSFERSET_NAME = "ORBID"
#TRANSFERSET_PATH = "../../data/orbid_testdata_0-10000_000.hdf5"
#TRANSFERSET_SUBSET_PATH = "orbid_subset.h5"
#TRANSFERSET_LABEL = [4]
#TEST_SIZE = 10000

#TRANSFERSET_NAME = "RADML"
#TRANSFERSET_PATH = "../../data/GOLD_XYZ_OSC.0001_1024.hdf5"
#TEST_SIZE = 2555904

''' Number of samples to work with '''
# when running train_test, needs to be 30 to match RadioML
# TODO change to be more flexible :)
#TEST_SIZE = 10000

''' Feature extraction '''
# higher_order features: 'element_HOS', 'RD_CTCF'
# second_order features: 'SCD', 'CHTC', 'CCSD'
FEATURE_TO_EXTRACT = "element_HOS"
EXTRACT_PROFILE = True
TRANSFERSET_FEATURESET_PATH = f"../../data/{TRANSFERSET_NAME}_{FEATURE_TO_EXTRACT}.h5"
SHAPE = (TEST_SIZE, 14)

''' ML parameters '''
# Options: LDA, SGD, LRG
CLASSIFIER = "LDA"
# True to use CUDA (not tested with changes), False otherwise
CUDA = False
