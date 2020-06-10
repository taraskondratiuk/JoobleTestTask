import sys

import numpy as np

import utility

batch_size = 100
num_of_features = 256

print(sys.argv[1])
print(sys.argv[3])
try:
    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]
except:
    raise Exception("no input or output file provided")

try:
    batch_size = int(sys.argv[3])
except:
    print("invalid batch size passed, set to 100")

try:
    num_of_features = int(sys.argv[4])
except:
    print("invalid num of features passed, set to 256")

input_file = open(input_file_name, "r")
input_file.readline()  # skipping header

lines_counter = 0

averages_for_batches = []

# counting averages
while True:
    # this approach skips last batch if its less than batch size
    # thats because batch size is usually much less than whole file and calculation error for average will be minimal
    # if maximum accuracy is needed we should count total lines and count
    # (total_lines_in_full_batches/total_lines * average_for_full_batches) +
    # (lines_in_last_batch/total_lines * average_for_last_batch)
    features_array = utility.get_batch_or_none(batch_size, num_of_features, input_file)

    if features_array is not None:
        averages_for_batches.append(features_array.mean(axis=0))  # average for every column of this batch
    else:
        break

averages = np.vstack(averages_for_batches).mean(axis=0)
input_file.close()

input_file = open(input_file_name, "r")
input_file.readline()

standard_deviations_for_batches = []

# counting standard deviations
while True:
    features_array = utility.get_batch_or_none(batch_size, num_of_features, input_file)

    if features_array is not None:
        features_minus_avg = features_array - averages
        standard_deviations_for_batches.append(
            np.abs(features_minus_avg))  # standard deviation for every column of this batch
    else:
        break

standard_deviations = np.vstack(standard_deviations_for_batches).mean(axis=0)
input_file.close()

input_file = open(input_file_name, "r")
input_file.readline()

output_file = open(output_file_name, "w")
output_file.write("id_job\t")
for i in range(num_of_features):
    output_file.write("feature_2_stand_" + str(i) + "\t")
output_file.write("max_feature_2_index\tmax_feature_2_abs_mean_diff\n")

# writing results
while True:
    line = input_file.readline()

    if not line:
        break

    list_of_cols = line.split()
    output_file.write(list_of_cols[0] + "\t")  # job id
    features = np.array([int(i) for i in list_of_cols[1].split(",")[1:]])
    stand_features = (features - averages) / standard_deviations

    output_file.write(",".join([str(i) for i in stand_features]))
    output_file.write("\t")

    index_of_max_el = np.argmax(features, axis=0)

    output_file.write(str(index_of_max_el) + "\t")
    output_file.write(str(abs(features[index_of_max_el] - averages[index_of_max_el])) + "\n")

output_file.close()
