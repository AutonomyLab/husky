#! /bin/bash

NPROC=0
for min_peak in 1 30 70 120
do
    for peak_sens in 3.5 3.625 3.75 3.875 4.0 4.125 4.25 4.375 4.5 4.625 4.75 4.875 5.0
    do
        for cluster_eps in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
        do
            for cluster_min_samples in 2 3 4 7
            do
                echo "min_peak $min_peak, peak_sens $peak_sens, cluster_eps $cluster_eps, cluster_min_samples $cluster_min_samples"

                frames="gesture_bags/lab/frames,gesture_bags/hallway/frames,gesture_bags/outside/frames,gesture_bags/null-outside/frames"
                anns="gesture_bags/lab/annotated3,gesture_bags/hallway/annotated3,gesture_bags/outside/annotated3,gesture_bags/null-outside/annotated"

                ./evaluate-detector.py -i $frames -a $anns --min_peak $min_peak --peak_sens $peak_sens --cluster_eps $cluster_eps --cluster_min_samples $cluster_min_samples 1>>tune.out &
                NPROC=$(($NPROC+1))
                if [ "$NPROC" -ge 8 ]; then
                    wait
                    NPROC=0
                fi
            done
        done
    done
done
