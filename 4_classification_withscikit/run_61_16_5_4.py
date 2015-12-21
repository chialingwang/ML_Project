
from sklearn.neighbors import KNeighborsClassifier
from sklearn import grid_search
import numpy as np
import re
import time
vlad_accee = "/scratch/cw2189/ML_SparsingModeling/3_get_vlad_withscikit_new"



def get_vlad(filename):

	f = open(filename)

	vlads = list()
	lables = list()
	for line in f.readlines():
        	result , lable = line.split(",")
        	vlad = list()
        	#print(result)
        	for item in result.split(" ") :
                	if re.findall("\d+\.\d+", item):
                        	vlad.append(float(item))
        	vlads.append(vlad)
        	lables.append(int(lable.replace('[','').replace(']','').replace('\n','')))
	f.close()
	return vlads , lables





def run(class_num, subsample_size , cluster_num, window_size,  method='knn' , n_nb = 2):
    
    # will load data as the patch size defined , 3 means 3*3 = 9 for each patch, and will return the dictionary included:
    # 'data'  (one patch)  , 'target' (the sample of this patch belongs to ) , 'filename' (the file comes from)
    bofs = []
    lable = []
    filename = "%s/TRAIN_VLAD_%d_%d_%d_%d.txt" %(vlad_accee , class_num , subsample_size , window_size , cluster_num)        
    bofs , lable = get_vlad(filename)

    #knn_init = KNeighborsClassifier()
    #parameters = {'n_neighbors':[ 5, 10 , 15]}
    #knn = grid_search.GridSearchCV(knn_init, parameters)

    bofs_test = []
    lable_test = []
    filename = "%s/TEST_VLAD_%d_%d_%d_%d.txt" %(vlad_accee , class_num , subsample_size , window_size , cluster_num)
    bofs_test , lable_test = get_vlad(filename)


    start = time.time()    
    if(method == "knn"):
        knn = KNeighborsClassifier(n_neighbors = n_nb)
        knn.fit(bofs, lable)
        predicted = knn.predict(bofs_test)
        score = knn.score(bofs_test,lable_test)
   
    print(time.time()-start) 

    return score  


if __name__ == "__main__":

    rnd_number    = 8131985
    class_num = 61
    subsample_size = 92
    window_size = 5
    cluster_num = 16
    n_nb = 4

    score = run(class_num, subsample_size , cluster_num, window_size,  method='knn' , n_nb = n_nb)

    print(score) 
