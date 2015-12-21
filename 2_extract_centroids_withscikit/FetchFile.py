'''

Homework 4

Name     : Tailin Lo
NetID    : tl1720
N number : N15116873
Email    : tl1720@nyu.edu

======================================

FetchFile Lib 

'''

import os, sys
import time
import image_norm_test as myData
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

pwd = os.pardir;
access = "../../database_patch";
filelist = []

def fetchFile(accessPath , numOfimageOfeachSample = 91):
    filelist = []
    index = 0
    i = 0
    for file in (sorted(os.listdir(accessPath))):
        if(index % 92 == i):
            completeName = os.path.join(accessPath, file)  
            filelist.append(completeName)
            if(i < numOfimageOfeachSample-1):  
                i+=1
            else:
                i = 0
        index +=1
    return filelist


def gen_data(class_num, subsample_size, window_size, rnd_number):
    accessPath = "%s/patchSize%d" %(access , window_size)
    filelist = fetchFile(accessPath , subsample_size)
    #print(filelist)
    image_num = class_num * subsample_size
    mydata = myData.load_data(filelist[0:image_num] , window_size)
    #print(mydata['data'][0:10])
    scaler = preprocessing.StandardScaler()
    x_norm = scaler.fit_transform(mydata['data'])
    return train_test_split(x_norm, mydata['target'], train_size=0.5, random_state=rnd_number)


if __name__ == "__main__":

    '''

        Unit Test for gen_data
        By uncomment the following code, you can run a simple KNN predict case (1 class/1 image/3 clusters)
        Note: 
        1. Those functions are used to fectch file from the folder "patch_database_non_normalize". Files in that folder are
        binary file. Every file includes a bunch of patch vectors for an image.

    '''
    '''
    rnd_number    = 8131985
    class_num = 1
    subsample_size = 1
    window_size = 3
    train_X, test_X, train_y, test_y = gen_data(class_num, subsample_size, window_size, rnd_number)
    print("Training Size: "+str(len(train_X)))
    '''
