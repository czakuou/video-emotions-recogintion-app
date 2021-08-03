from os import path

# base path to the emotion dataset
BASE_PATH = "./datasets/fer2013"

# path to the input emotions file
INPUT_PATH = path.sep.join([BASE_PATH, "fer2013/fer2013.csv"])

NUM_CALSSES = 6

# paths to training, testing and validation HDF5 files
TRAIN_HDF5 = path.sep.join([BASE_PATH, "hdf5/train.hdf5"])
VAL_HDF5 = path.sep.join([BASE_PATH, "hdf5/val.hdf5"])
TEST_HDF5 = path.sep.join([BASE_PATH, "hdf5/test.hdf5"])

BATCH_SIZE = 256

# output path for logs
OUTPUT_PATH = path.sep.join([BASE_PATH, "output"])

INIT_LR = 1e-5
MAX_LR = 1e-2