import os

# Running mode
local_mode = True

# Local dir
local_dir = '?'

# Remote dirs
tmp_dir = '?'


def get_base_dir():
    if local_mode:
        return local_dir
    else:
        return tmp_dir


base_dir = get_base_dir()

if not os.path.exists(base_dir):
    raise FileNotFoundError(base_dir)

acoustic_dir = '?'
semantic_dir = '?'

# Note: csv files are not provided in this repository.
train_partition_csv = os.path.join(acoustic_dir, 'fold0.csv')
valid_partition_csv = os.path.join(acoustic_dir, 'fold1.csv')
test_partition1_csv = os.path.join(acoustic_dir, 'fold2.csv')
test_partition2_csv = os.path.join(acoustic_dir, 'fold3.csv')
test_partition3_csv = os.path.join(acoustic_dir, 'fold4.csv')

embedding_npz = os.path.join(semantic_dir, 'WLE.npy')
# embedding_npz = os.path.join(semantic_dir, 'RANDOM_EMB.npy')

label_npz = os.path.join(semantic_dir, 'index_label.npy')

num_acoustic = 10
acoustic_features = 128
semantic_features = 300
