'''

Homework 5

Name     : Tailin Lo
NetID    : tl1720
N number : N15116873
Email    : tl1720@nyu.edu
 
'''
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import svm
from collections import deque
import numpy as np
import math
import matplotlib.pyplot as plt
import SVM
import time

# ============================================= end ============================================= #	

def norm_wo_root(v):
	sum = 0
	for vi in v:
		sum += (vi*vi)
	# end for
	return sum

# ============================================= end ============================================= #	

def gen_line_points(x, w) :
	line_points = list()
	y = list()
	for xi in x:
		#yi = -w[0]/w[1]*xi 
		yi = -(w[0]*xi+w[2])/w[1]
		y.append(yi)
	# end for
	line_points.append(x)
	line_points.append(y)
	return line_points

# ============================================= end ============================================= #	

def dataset_fixed_cov(plus_size, minus_size, dim, dist, rnd_number):
	np.random.seed(rnd_number)
	w = np.random.randint(100, size=(1, 5)).flatten()
	C = np.random.normal(0, 1, (dim, dim))
	X = np.r_[np.dot(np.random.randn(minus_size, dim), C), np.dot(np.random.randn(plus_size, dim), C) + dist*np.ones(dim)]
	y = np.hstack((np.multiply(-1, np.ones(minus_size)) , np.ones(plus_size)))

	return X, y

# ============================================= end ============================================= #	

def data_bias_handle(X) :
	XX = []
	for i in range(len(X)):
		x = np.append(X[i], [1])
		XX.append(x)
	# end for
	return np.array(XX)


# ============================================= end ============================================= #	

def gen_validation_data(X, y, validation_size) :
	total_size = len(X)
	train_index = total_size - validation_size
	train_X = X[0:train_index]
	train_y = y[0:train_index]
	validation_X = X[train_index:total_size]
	validation_y = y[train_index:total_size]
	return train_X, train_y, validation_X, validation_y

# ============================================= end ============================================= #	

def calc_error(predict, target) :
	error = 0
	for i in range(len(target)) :
		if target[i]*predict[i] < 0:
			error += 1
		# end if
	# end for
	return error/len(target), error
	
# ============================================= end ============================================= #	

def show_data_figure(x, y, line_points) :
	color = ["#FF0000", "#00FF00", "#0000FF", "#DB7093", "#EEE8AA", "#FFEFD5", "#DDA0DD", "#9ACD32", "#F5DEB3", "#008080"]
	fig = plt.figure(figsize=(15, 10))
	for i in range(len(x)) :
		plt.scatter(x[i][0] , x[i][1] , c=color[int(y[i])+1], s=100)
	# end for
	plt.plot(line_points[0], line_points[1])
	plt.show()

# ============================================= end ============================================= #	

def show_obj_versus_iteration(step_list, obj_list) :
	x = list()
	for i in range(1, len(obj_list[0])+1) :
		x.append(i)
	# end for
	fig = plt.figure(figsize=(15, 10))
	fig.suptitle("Object Value V.S. Iteration" , fontsize=16) 
	for i in range(len(obj_list)) :
		label_str = "step: " + str(step_list[i])
		plt.plot(x, obj_list[i], label=label_str)
		plt.legend(loc=1)
	# end for
	plt.show()

# ============================================= end ============================================= #	

def show_obj_versus_iteration_2(para_list, obj_list) :
	x = list()
	for i in range(1, len(obj_list[0])+1) :
		x.append(i)
	# end for
	fig = plt.figure(figsize=(15, 10))
	fig.suptitle("Object Value V.S. Iteration" , fontsize=16) 
	for i in range(len(obj_list)) :
		label_str = "eta0:" + str(para_list[i][0]) + " , t0:" + str(para_list[i][1])
		plt.plot(x, obj_list[i], label=label_str)
		plt.legend(loc=1)
	# end for
	plt.show()

# ============================================= end ============================================= #	

def show_train_test_error_versus_iteration(train_error_list, test_error_list) :
	x = list()
	for i in range(1, len(train_error_list)+1) :
		x.append(i)
	# end for
	fig = plt.figure(figsize=(15, 10))
	fig.suptitle("Error V.S. Iteration" , fontsize=16) 

	#plt.subplot(2, 1, 0)
	plt.plot(x, train_error_list, label="train error")
	plt.legend(loc=1)
	#plt.subplot(2, 1, 1)
	plt.plot(x, test_error_list, label="test error")
	plt.legend(loc=1)
	plt.show()

# ============================================= end ============================================= #	

def show_error_versus_iteration(step_list, error_list) :
	fig = plt.figure(figsize=(15, 10))
	#fig.suptitle("Object Value V.S. Iteration" , fontsize=16) 
	plt.plot(step_list, error_list)
	plt.show()

# ============================================= end ============================================= #	

def split_data(split_num, X, y) :
	split_data_list = list()
	data_size = len(y)
	split_size = int(data_size/split_num)
	#print(data_size," , ",split_size)
	for i in range(split_num):
		split_data_list.append((X[i*split_size:(i+1)*split_size], y[i*split_size:(i+1)*split_size]))
	# end for
	return 	split_data_list

# ============================================= end ============================================= #	

def cross_validation(point_dim, split_num, split_data_list, classifier, mymethod=False) :
	validation_error = 0
	for i in range(split_num) :
		if mymethod :
			classifier.validation_X = split_data_list[i][0]
			classifier.validation_y = split_data_list[i][1]
		# end if

		# merge remaining dataset
		train_X = np.array([np.zeros(point_dim)])
		train_y = np.array([0])
		for j in range(split_num) :
			if j != i :
				#print("j: ",j," , ",split_data_list[j][0])
				train_X = np.concatenate((train_X, split_data_list[j][0]))
				train_y = np.concatenate((train_y, split_data_list[j][1]))
			# end if
		# end for

		train_X = np.delete(train_X, 0, 0)
		train_y = np.delete(train_y, 0, 0)

		# train data and predict
		#print(i," , ",len(train_X)," , ",len(train_y))
		classifier.fit(train_X, train_y)
		predict = classifier.predict(split_data_list[i][0])
		error_rate, error_num = calc_error(predict, split_data_list[i][1])
		validation_error += error_rate
	# end for
	return validation_error/split_num
	

# ============================================= end ============================================= #	

# grid search for both mysvm and sklearn svm
def grid_search_for_C(c_seq, X, y, split_num, myclassifier) :

	train_X      = []  
	train_y      = []     
	validation_X = []
	validation_y = []

	train_X.append(myclassifier.train_X)      
	train_y.append(myclassifier.train_y)  
	validation_X.append(myclassifier.validation_X)
	validation_y.append(myclassifier.validation_y)

	dim = len(X[0])
	split_data_list = split_data(split_num, X, y)
	min_error1 = 1
	min_error2 = 1
	min_C1 = 1
	min_C2 = 1
	for c in c_seq :
		myclassifier.C = c
		clf = svm.SVC(kernel='linear', C=c)
		
		# search mysvm
		validation_error = cross_validation(dim, split_num, split_data_list, myclassifier, True)
#		print("C :" , c, "Error :" validation_error) 
		if min_error1 > validation_error :
			min_error1 = validation_error
			min_C1 = c
		# end if
		
		# search sklearn svm
		validation_error = cross_validation(dim, split_num, split_data_list, clf, True)
		if min_error2 > validation_error :
			min_error2 = validation_error
			min_C2 = c
		# end if
	# end for

	# recover the original set
	myclassifier.train_X      = train_X[0]
	myclassifier.train_y      = train_y[0]
	myclassifier.validation_X = validation_X[0]
	myclassifier.validation_y = validation_y[0]

	return min_C1, min_C2, min_error1, min_error2

# ============================================= end ============================================= #	

# grid search for both mysvm and sklearn svm
def grid_search_for_C_2(c_seq, X, y, split_num, myclassifier) :

	train_X      = []  
	train_y      = []     
	validation_X = []
	validation_y = []

	train_X.append(myclassifier.train_X)      
	train_y.append(myclassifier.train_y)  
	validation_X.append(myclassifier.validation_X)
	validation_y.append(myclassifier.validation_y)

	dim = len(X[0])
	split_data_list = split_data(split_num, X, y)
	min_error1 = 1
	min_C1 = 1
	for c in c_seq :
		myclassifier.C = c
		
		# search mysvm
		validation_error = cross_validation(dim, split_num, split_data_list, myclassifier, True)
		if validation_error == 0:
			continue
		if min_error1 > validation_error :
			min_error1 = validation_error
			min_C1 = c
		# end if
		
	# end for

	# recover the original set
	myclassifier.train_X      = train_X[0]
	myclassifier.train_y      = train_y[0]
	myclassifier.validation_X = validation_X[0]
	myclassifier.validation_y = validation_y[0]

	return min_C1, min_error1

# ============================================= end ============================================= #	
'''
alpha			= 0.2
beta			= 0.8
C 				= 1
h 				= 0.5
max_iter 		= 100
step_size 		= 1
group_distance  = 1
unbalance_ratio = 0.75
total_size 		= 1000
plus_size		= int(total_size*unbalance_ratio)
minus_size      = total_size - plus_size
dimension 		= 2
train_test_ratio = 0.5
validation_size = int(total_size * train_test_ratio * 0.1)
gradient_error  = 1e-3
improve_ratio   = 0.9
option     	    = 2
rnd_number      = 8131985



################################################################
#	Data Generation (Unbalanced)
################################################################
StandardScaler = preprocessing.StandardScaler()
rng = np.random.RandomState(rnd_number)
X, y = dataset_fixed_cov(plus_size, minus_size, dimension, group_distance, rnd_number)
X = StandardScaler.fit_transform(X)
permutation = rng.permutation(len(X))
X, y = X[permutation], y[permutation]
X = data_bias_handle(X)
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=train_test_ratio, random_state=rnd_number)
train_XX, train_yy, validation_XX, validation_yy = gen_validation_data(train_X, train_y, validation_size)
#print(len(train_y)," , ",len(test_y)," , ",len(validation_y))
'''

'''

	Section 2.1.4

	plot objection function value v.s. iteration with different step size

	summary:
	1. When step size becomes large, the speed of converge is increasing. The optimized step size is about
	   0.4 to 0.6.
	2. When step size is 0.672, the objection value oscillates. But there still exists a stable condition.
	3. When step size is 0.814, the objection value oscillates. But there still doesn't exists a stable condition.
	4. Although the speed of converge is increasing by increasing step size, this method may cause the search will
	   jump outside the velly, which means it can't get the minimum point. 

'''
'''
mysvm = SVM.MySVM(alpha, beta, C, h, max_iter, step_size, rnd_number, train_XX, train_yy, validation_XX, validation_yy, gradient_error, improve_ratio)
obj_list = list()
step_list = list()
mysvm.max_iter = 20
for k in [2*i for i in range(12)] :
	mysvm.step_size = 0.1*math.pow(1.1,k)
	mysvm.my_gradient_decent(1, True)
	step_list.append(mysvm.step_size)
	obj_list.append(mysvm.obj_record)
# end for
show_obj_versus_iteration(step_list, obj_list)

'''
'''

	Section 2.1.4

	plot train error and test error v.s. iteration with a fixed step size

	summary:

	1. I get the optimized train error when the iteration is 5
	2. I get the optimized test error when the iteration is 5
	3. From the above analysis, we can know the best step size is about 0.5 

'''
'''
train_error_list = list()
test_error_list  = list()
mysvm.step_size = 0.1*math.pow(1.1,10)
for iter in range(1,21) :
	mysvm.max_iter = iter
	mysvm.my_gradient_decent(1, True)
	predict_train = mysvm.predict(train_X)
	predict_test  = mysvm.predict(test_X)
	train_error_rate, train_error_num = calc_error(predict_train, train_y)
	test_error_rate, test_error_num = calc_error(predict_test, test_y)
	train_error_list.append(train_error_rate)
	test_error_list.append(test_error_rate)
# end for
show_train_test_error_versus_iteration(train_error_list, test_error_list)

'''
'''

	Section 2.1.4

	fixed iteration number, and plot error v.s. step size

	summary:

	1. From the figure, we can know the best step size is about 0.55


'''
'''
mysvm.max_iter = 6
train_error_list = list()
test_error_list  = list()
step_list = list()
for k in [2*i for i in range(12)] :
	mysvm.step_size = 0.1*math.pow(1.1,k)
	mysvm.my_gradient_decent(1, True)
	step_list.append(mysvm.step_size)
	predict_train = mysvm.predict(train_X)
	predict_test  = mysvm.predict(test_X)
	train_error_rate, train_error_num = calc_error(predict_train, train_y)
	test_error_rate, test_error_num   = calc_error(predict_test, test_y)
	train_error_list.append(train_error_rate)
	test_error_list.append(test_error_rate)
# end for
#show_error_versus_iteration(step_list, test_error_list)

'''
'''

	Section 2.1.4

	Fix step size to be 0.55, and compare time of fixed step size with time of backtracking line search
	condtion: no stop criterion, i.e. run 50 iterations

	summary:

	1. Excution time of fixed step size is about 0.08 sec, and excution time of backtracking line search is 0.1 sec.
	   Thus, backtracking line search will take (0.1-0.08)/50 = 0.4 ms at each iteration.


'''
'''
mysvm.max_iter = 50
mysvm.step_size = 0.55
start = time.time()
mysvm.my_gradient_decent(1, True)
finish = time.time()
print("Execution Time(fixed step size): ",finish-start," sec")
start = time.time()
mysvm.my_gradient_decent(1, False)
finish = time.time()
print("Execution Time(backtracking line search): ",finish-start," sec")

'''
'''

	Section 2.1.4

	Cross validation to find optima C
	Condition: 1. 10 fold cross validation 
	           2. search by power of 2 from -10 to 10

	summary:

	1. By cross validation, both mysvm and sklearn svm almost have the same error rate, i.e 3.4%, with group_distance = 5
	2. Modify parameter "group_distance" to 1(very strong overlap), both mysvm and sklearn svm almost 
	   have the same error rate, i.e 24.6%

'''
'''
min_C1, min_C2, min_error1, min_error2 = grid_search_for_C([2**i for i in range(-10,10) ], train_X, train_y, 10, mysvm)
print(min_C1," , ",min_C2," , ",min_error1," , ",min_error2)

mysvm.max_iter = 1000
mysvm.C = min_C1
mysvm.my_gradient_decent(3)
print("w: ",mysvm.w)
predict_test = mysvm.predict(test_X)
test_error_rate, test_error_num = calc_error(predict_test, test_y)
print("-- My SVM -- optima C: ",min_C1," , general error rate: ",test_error_rate," , general error number",test_error_num)


clf = svm.SVC(kernel='linear', C = min_C2)
clf.fit(train_X, train_y)
predict_test = clf.predict(test_X)
test_error_rate, test_error_num = calc_error(predict_test, test_y)
print("-- Sklearn SVM -- optima C: ",min_C2," , general error rate: ",test_error_rate," , general error number",test_error_num)

'''
'''
################################################################
#	Big Data Generation (Unbalanced)
################################################################
total_size 		= 1e4
plus_size		= int(total_size*unbalance_ratio)
minus_size      = total_size - plus_size
X, y = dataset_fixed_cov(plus_size, minus_size, dimension, group_distance, rnd_number)
X = StandardScaler.fit_transform(X)
permutation = rng.permutation(len(X))
X, y = X[permutation], y[permutation]
X = data_bias_handle(X)
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=train_test_ratio, random_state=rnd_number)
train_XX, train_yy, validation_XX, validation_yy = gen_validation_data(train_X, train_y, validation_size)
'''

'''

	Section 2.2

	Big Data by Gradient Decent 

	summary: 

	1. excution time is 313.1292259693146 sec


'''
'''
mysvm = SVM.MySVM(alpha, beta, C, h, max_iter, step_size, rnd_number, train_XX, train_yy, validation_XX, validation_yy, gradient_error, improve_ratio)
mysvm.max_iter = 100
start = time.time()
mysvm.my_gradient_decent(3)
finish = time.time()
print("Gradient Decent Big Data Execution Time: ",finish-start," sec")
'''

'''

	Section 2.2

	plot objection value with different eta0 and t0

	summary:
	1. Although there are different initial objection value with different eta0 and t0, 
	   these objection values still reach the stable values 

	2. Small eta0 and t0 will reach stable value fast

	3. When there is a stop criterion, use SGD when data is large, and use GD when data is small 

	4. GD always generates the same result, but SGD can't


'''
'''

subsample_size = 500
mysvm = SVM.MySVM(alpha, beta, C, h, max_iter, step_size, rnd_number, train_XX[0:subsample_size], train_yy[0:subsample_size], \
	              validation_XX, validation_yy, gradient_error, improve_ratio)
mysvm.max_iter = 50
print(mysvm.train_size)
obj_list = list()
para_list = list()
mysvm.eta0 = 1.0

start = time.time()
#for eta0 in [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0] :
for eta0 in [1.0, 1.25, 1.5] :
	for t0 in range(1,5) :
		mysvm.eta0 = eta0
		mysvm.t0   = t0
		para_list.append((mysvm.eta0, mysvm.t0))
		mysvm.my_sgd(1,1)
		obj_list.append(mysvm.obj_record)
# end for
#finish = time.time()
print("Execution Time: ",finish-start," sec")
show_obj_versus_iteration_2(para_list, obj_list)

'''
'''

	Section 2.2

	Grid Search for different n, d, C

	Summary:


'''
'''
group_distance = 1
unbalance_ratio = 0.6
#for n in [1e3, 1e4, 1e5] :
count = 0

for n in [1e3, 1e4, 1e5] :
	#for dim in [2**i for i in range(1,11)] :
	#for dim in [2**i for i in [1, 2, 6, 8 ,10]] :
	for dim in [2**i for i in [1]] :
		start = time.time()
		total_size = n
		plus_size  = int(total_size*unbalance_ratio)
		minus_size = total_size - plus_size
		X, y = dataset_fixed_cov(plus_size, minus_size, dim, group_distance, count)
		X = StandardScaler.fit_transform(X)
		permutation = rng.permutation(len(X))
		X, y = X[permutation], y[permutation]
		X = data_bias_handle(X)
		train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=train_test_ratio, random_state=rnd_number)
		train_XX, train_yy, validation_XX, validation_yy = gen_validation_data(train_X, train_y, validation_size)

		mysvm = SVM.MySVM(alpha, beta, C, h, max_iter, step_size, rnd_number, train_XX, train_yy, \
	              		  validation_XX, validation_yy, gradient_error, improve_ratio)
		mysvm.max_iter = 1000
		min_C1, min_error1 = grid_search_for_C_2([2**i for i in range(-10,10)], train_XX, train_yy, 10, mysvm)
		predict_test = mysvm.predict(test_X)
		print(len(predict_test)," , ",len(test_y)," , ",len(test_X)," , ",len(mysvm.w))
		test_error_rate, test_error_num = calc_error(predict_test, test_y)
		finish = time.time()
		print("n: ",n," , d: ",dim," , C: ",min_C1," , error rate: ",test_error_rate," , error num: ",test_error_num)		
		print("Execution Time: ",finish-start," sec")
		count += 1
	# end for
# end for
'''






