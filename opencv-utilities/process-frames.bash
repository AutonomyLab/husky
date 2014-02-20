#! /bin/bash

NPROC=0
mkdir created-videos
cd created-videos

for dataset in lab hallway outside null-outside
do
    mkdir $dataset
    cd $dataset

    for mode in {0..24}
    do
        mkdir $mode
        cd $mode
        inner=`pwd`
        
        ~/periodic/image-tools.py --save $inner -i ~/periodic/gesture_bags/$dataset/frames -m $mode --headless &

        cd ..

        NPROC=$(($NPROC+1))
        if [ "$NPROC" -ge 8 ]; then
            wait
            NPROC=0
        fi
    done

    cd ..
done
