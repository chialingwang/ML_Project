#!/bin/bash
class_num=61
echo $class_num
window=5
for i in 6
#for i in 4 16 24 32 40 48 56 64 128 512;
do
	cp run_script.py run_script_${class_num}_${i}_${window}.py;
        sed -i "s/__CLASS_NUM__/$class_num/g"  run_script_${class_num}_${i}_${window}.py;
	
	sed -i "s/__CLUSTER_NUM__/$i/g"  run_script_${class_num}_${i}_${window}.py;
        sed -i "s/__WINDOW_SIZE__/$window/g"  run_script_${class_num}_${i}_${window}.py;

	cp run_centroids.pbs run_centroids_${class_num}_${i}_${window}.pbs;
	sed -i "s/get_vlad__/gvlad_${class_num}_${i}_${window}/g" run_centroids_${class_num}_${i}_${window}.pbs;
	sed -i "s/run_script/run_script_${class_num}_${i}_${window}/g" run_centroids_${class_num}_${i}_${window}.pbs;
	qsub run_centroids_${class_num}_${i}_${window}.pbs;
done

