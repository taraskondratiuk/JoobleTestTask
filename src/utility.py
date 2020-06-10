import numpy as np


def get_batch_or_none(batch_size, num_of_features, file):
    features_array = np.empty(shape=(batch_size, num_of_features), dtype=int)
    for i in range(0, batch_size):
        line = file.readline()

        if not line:
            return None

        features_array[i] = [int(i) for i in line.split()[1].split(",")[1:]]

    return features_array
