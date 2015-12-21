import os, sys
import image_norm_test as myData;
import numpy as np
from sklearn.cross_validation import train_test_split

import re
import FetchFile
import get_feature
import time
import run_vlad

centroid_accee = "/scratch/cw2189/ML_SparsingModeling/2_extract_centroids_withscikit"

def got_feature(class_num, subsample_size, window_size, cluster_num_list):
    centroid = []
    count = 0
    temp = []
    n = window_size*window_size	
    for cluster in cluster_num_list:
        centroid_file = "%s/%d_%d_%d_%d.txt" %(centroid_accee, class_num, subsample_size, window_size, cluster)
        centroid.append(get_feature.read_features(centroid_file))
    return centroid


if __name__ == "__main__":

    rnd_number    = 8131985
    class_num = __CLASS_NUM__
    subsample_size = 92
    window_size = __WINDOW_SIZE__
    cluster_num = [__CLUSTER_NUM__]

###################################################
# permutation data and split into train and test #
# split our data into half of train data and half of text data randomly.
train_X, test_X, train_y, test_y = FetchFile.gen_file(class_num, subsample_size, window_size, rnd_number)
##  USAGE :  gen_file(class_num, subsample_size, window_size, rnd_number)  ###  


print(train_y)
print(test_y)

###################################################
#Start to get the features learned from feed in data

centroid = got_feature(class_num, subsample_size, window_size, cluster_num)

print(centroid[0])
print(len(centroid[0]))

###################################################
# Get VLAD of Each Image
bofs = []
bofs_test = []
start = time.time()
#bofs , bofs_test = run_vlad.run(class_num, 3 , centroid, cluster_num, 0 , window_size , train_X[0:2] , train_y[0:2], test_X[0:2] , test_y[0:2])
bofs , bofs_test = run_vlad.run(class_num, subsample_size , centroid, cluster_num, 0 , window_size , train_X , train_y, test_X , test_y )
print("Get VLAD: ",time.time()-start)





