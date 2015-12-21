#!/bin/bash


module load scikit-learn/intel/0.17b1
module load  zlib/intel/1.2.8
module load pillow/intel/2.7.0


class_num=61
echo $class_num
window=5
echo $window
for i in 6
#for i in 4 16 24 32 40 48 64 128 512;  # cluser_num
do	
	echo "CLUST : $i"
	for neibor in 1 4 16 32;
	do
		echo "NEIBOR NUM : $neibor"
		cp run2.py run_${class_num}_${i}_${window}_${neibor}.py;
        	sed -i "s/__CLASS_NUM__/$class_num/g"   run_${class_num}_${i}_${window}_${neibor}.py;

		sed -i "s/__CLUST_NUM__/$i/g"   run_${class_num}_${i}_${window}_${neibor}.py;
		sed -i "s/__WINDOW__/$window/g"   run_${class_num}_${i}_${window}_${neibor}.py;

		sed -i "s/__NUM_NEIBOR__/$neibor/g"   run_${class_num}_${i}_${window}_${neibor}.py;
		python run_${class_num}_${i}_${window}_${neibor}.py >> score_${class_num}_${i}_${window}_${neibor}.log
	done
done

