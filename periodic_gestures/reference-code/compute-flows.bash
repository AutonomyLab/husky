#! /bin/bash

# use detect-periodic.py to run through all our training videos
# and identify regions of periodic motion, and export the flow
# values for training a classifier


for dataset in tshirt sweater no-gestures
do

# dataset=$1

for size in 50m 25m 20m 15m 10m
do

# size=$2

~/ml-project/detect-periodic.py ./$dataset/$size --outdir ./$dataset/output-$size --headless > /dev/null &
# echo "tar zcf ./$dataset-o-$size.tgz ./$dataset/output-$size"
# tar zcf ./$dataset-o-$size.tgz ./$dataset/output-$size
# rm -r ./$dataset/output-$size

done
done
