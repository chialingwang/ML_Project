#!/bin/bash


module load scikit-learn/intel/0.17b1
module load  zlib/intel/1.2.8
module load pillow/intel/2.7.0


class_num=61
echo $class_num
window=3
echo $window
#for i in 4
for i in 16 32 64 128 512 1024;  # cluser_num
do	
	echo "CLUST : $i"
	for neibor in 1 4 16 32 64 128;
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

