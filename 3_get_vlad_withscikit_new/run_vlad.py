from my_vlad import my_vlad
from sklearn.neighbors import KNeighborsClassifier
from sklearn import grid_search
import numpy as np
import image_norm_test as myData;


def get_vlad(FileList , filename , window_size , vlad):
    bofs = []
    f = open(filename, 'w+')

    for file in FileList:
        mydata,y = myData.load_sig_data(file , window_size)
        vlad_result = vlad.get_vlad(mydata['data']).flatten() 
        bofs.append(vlad_result)
	for each in vlad_result:
		f.writelines(str(each)+" ")
	f.writelines(","+ str(y))
        f.writelines("\n")
    f.close()
    return bofs

def run(class_num, subsample_size , centroid , cluster_num , group_num , window_size ,  train_X , train_y ,test_X , test_y):
    
    # will load data as the patch size defined , 3 means 3*3 = 9 for each patch, and will return the dictionary included:
    # 'data'  (one patch)  , 'target' (the sample of this patch belongs to ) , 'filename' (the file comes from)
    bofs = []
    vlad = my_vlad(centroid[group_num])
    filename = "TRAIN_VLAD_%d_%d_%d_%d.txt" %(class_num, subsample_size , window_size, cluster_num[group_num])
    bofs=get_vlad(train_X, filename , window_size, vlad)   

    bofs_test = []
    filename = "TEST_VLAD_%d_%d_%d_%d.txt" %(class_num, subsample_size , window_size, cluster_num[group_num])

    bofs_test=get_vlad(test_X, filename , window_size, vlad)


    print(len(bofs))

    return bofs , bofs_test    
