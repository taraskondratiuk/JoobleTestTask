## How to run
The app uses docker so you can:
```
docker build -t <IMAGE_NAME> .

docker run -it -v <PATH_TO_MOUNT_DIR>/data:/data <NAME_OF_INPUT_FILE_IN_DATA_DIR> <NAME_OF_OUTPUT_FILE> <BATCH_SIZE> <NUM_OF_FEATURES>
```
BATCH_SIZE, NUM_OF_FEATURES - optional

On windows works only with cmd

To run with python you should:
```
pip3 install numpy

python src/main_job.py <PATH_TO_INPUT_FILE> <PATH_TO_OUTPUT_FILE> <BATCH_SIZE> <NUM_OF_FEATURES>
```
BATCH_SIZE, NUM_OF_FEATURES - optional