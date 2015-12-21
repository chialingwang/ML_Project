#!/bin/bash

class_num=2
window=5
echo $class_num
echo $window
#for i in 512 1024 
for i in  8;
do
	cp GridSearch.py GridSearch_${class_num}_${i}_${window}.py;
        sed -i "s/__CLASS_NUM__/$class_num/g"  GridSearch_${class_num}_${i}_${window}.py;
	sed -i "s/__CLUSTER_NUM__/$i/g"  GridSearch_${class_num}_${i}_${window}.py;
        sed -i "s/__WINDOW__/$window/g"  GridSearch_${class_num}_${i}_${window}.py;

        cp run_centroids.pbs run_centroids_${class_num}_${i}.pbs;
	sed -i "s/centrolize/sciCen_${class_num}_${i}_${window}/g" run_centroids_${class_num}_${i}.pbs;
	sed -i "s/GridSearch/GridSearch_${class_num}_${i}_${window}/g" run_centroids_${class_num}_${i}.pbs;
	qsub run_centroids_${class_num}_${i}.pbs;
done

