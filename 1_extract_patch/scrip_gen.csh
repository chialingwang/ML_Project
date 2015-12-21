#!/bin/bash


for i in 7
#for i in 16 32 64 128 512;
do
	cp image_patch_process.py image_patch_process_$i.py;
        sed -i "s/__PATCH_SIZE__/$i/g"  image_patch_process_$i.py;
        cp  run_patch.pbs run_patch_${i}.pbs;
	sed -i "s/__PATCH_SIZE__/${i}/g" run_patch_${i}.pbs;
	qsub run_patch_${i}.pbs;
done

