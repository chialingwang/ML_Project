
from sklearn.neighbors import KNeighborsClassifier
from sklearn import grid_search
import numpy as np
import re
import Choose_target
from sklearn import svm
import SVM
import runGrid as Grid
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





def run(class_num, subsample_size , cluster_num, window_size  ):
    
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
   
    TRAIN = Choose_target.Choose_Target(bofs,lable)
    TRAIN.select(1)
    
    TEST = Choose_target.Choose_Target(bofs_test,lable_test)
    TEST.select(1)
    
    return  TRAIN.X , TRAIN.y ,  TEST.X , TEST.y


if __name__ == "__main__":

    rnd_number    = 8131985
#    score = run(class_num, subsample_size , cluster_num, window_size,  method='knn' , n_nb = n_nb)

    class_num = 2
    subsample_size = 92
    window_size = 5
    cluster_num = 8


    alpha                   = 0.2
    beta                    = 0.8
    C                               = 1
    h                               = 0.5
    max_iter                = 100
    step_size               = 1
    gradient_error  = 1e-3
    improve_ratio   = 0.9
    option              = 2
    validation_size = int(class_num * subsample_size * 0.5 * 0.1)

    Train_X , Train_y , Test_X , Test_y = run(class_num, subsample_size , cluster_num, window_size)
#    print(Train_y , Test_y)
    Train_X = np.asarray(Train_X)
    Train_y = np.asarray(Train_y)
    Test_X = np.asarray(Test_X)
    Test_y = np.asarray(Test_y)

    Train_XX, Train_yy, validation_XX, validation_yy = Grid.gen_validation_data(Train_X,Train_y,validation_size)

#    for i in range(5,10):
for i in range(-3,10 , 1):
	C = 2**i
    	mysvm = SVM.MySVM(alpha, beta, C, h, max_iter, step_size, rnd_number, Train_XX, Train_yy, validation_XX, validation_yy, gradient_error, improve_ratio)

    #min_C1, min_C2, min_error1, min_error2 = Grid.grid_search_for_C([2**i for i in range(-10,10) ], Train_X, Train_y, 10, mysvm)
    #print(min_C1," , ",min_C2," , ",min_error1," , ",min_error2)
    	start = time.time()
   	mysvm.fit(Train_X , Train_y) 
    	predict_test = mysvm.predict(Test_X)
#        print(mysvm.w)
#	print(predict_test)
#	print(Test_y)

    	test_error_rate, test_error_num = Grid.calc_error(predict_test, Test_y)

    	finish = time.time()

    	print("-- My SVM -- optima C: ",C," , general error rate: ",test_error_rate," , general error number",test_error_num)
    	print("Execution Time: ",finish-start," sec")
    	
	#start = time.time()

    	#clf = svm.SVC(kernel='linear', C = min_C2)
    	#clf.fit(Train_X, Train_y)
    	#predict_test = clf.predict(Test_X)
    	#test_error_rate, test_error_num = calc_error(predict_test, Test_y)
    	#finish = time.time()

    	#print("-- Sklearn SVM -- optima C: ",min_C2," , general error rate: ",test_error_rate," , general error number",test_error_num)
    	#print("Execution Time: ",finish-start," sec")


#    print(score) 
