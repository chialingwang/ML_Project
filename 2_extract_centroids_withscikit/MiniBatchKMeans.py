'''

Homework 5

Name     : Tailin Lo
NetID    : tl1720
N number : N15116873
Email    : tl1720@nyu.edu

'''
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn import preprocessing
from collections import deque


# ============================================= end ============================================= #	

class MiniBatchKMeans :
	def __init__(self, data_set, max_iter, cluster_num, mini_batch_size) :
		self.max_iter        = max_iter
		self.data_set        = data_set
		self.cluster_num     = cluster_num
		self.mini_batch_size = mini_batch_size
		self.data_size       = len(data_set)
		self.min_means       = list()
		self.means           = list()
		self.bound_matrix    = list()
		self.obj_record      = list()
		self.min_obj         = 1e12


	# ============================== end ============================== #	

	def object_value_calc(self):
		sum = 0.0
		for i in range(self.data_size) :
			diff = self.data_set[i] - self.means[self.bound_matrix[i][2]]
			sum += np.dot(diff, diff)
		# end for
		return sum

	# ============================== end ============================== #	

	def means_points_distance_matrix(self) :
		mp_dist = list()
		for data in self.data_set:
			dist = list()
			for mean in self.means:
				diff = data - mean
				dist.append(np.dot(diff, diff))
			# end for
			mp_dist.append(dist)
		# end for
		return mp_dist

	# ============================== end ============================== #	

	def means_means_distance_matrix(self, scale=0.5) :
		mm_dist = list()
		mm_closeest_dist = list()
		for i in range(self.cluster_num):
			dist = list()
			for j in range(self.cluster_num):
				if i == j: 
					dist.append(1e12)
				elif i < j:
					diff = self.means[i] - self.means[j]
					dist.append(scale*np.dot(diff, diff))
				else:
					dist.append(mm_dist[j][i])
				# end if
			# end for
			mm_dist.append(dist)
		# end for

		for k in range(self.cluster_num):
			mm_closeest_dist.append(min(mm_dist[k]))
		# end for
		return mm_closeest_dist, mm_dist

	# ============================== end ============================== #	

	def get_closest_cluster(self, point) :
		min_val = 1e12
		min_k   = -1
		for k in range(len(self.means)) :
			diff = self.means[k] - point;
			dist = np.dot(diff, diff)
			if min_val > dist :
				min_val = dist
				min_k   = k
			# end if
		# end for
		return min_k, min_val

	# ============================== end ============================== #	

	def run(self) :
		window_obj = deque()
		window_size = 50	
		self.min_means    = list()
		self.means        = list()
		self.bound_matrix = list()
		self.obj_record   = list()
		self.min_obj      = 1e12

		# initialize cluster
		# the method is the same as mykmeams++
		self.means.append(self.data_set[0])
		for i in range(1, self.cluster_num):
			dist_list = list()
			acc_dist = 0.0
			for j in range(self.data_size) :
				k, dist = self.get_closest_cluster(self.data_set[j])
				acc_dist += dist
				dist_list.append((j, acc_dist))
			# end for 
			choose_num = acc_dist*np.random.random_sample()

			for i in range(len(dist_list)) :
				if choose_num > dist_list[i][1] :
					continue
				else :
					self.means.append(self.data_set[dist_list[i][0]])
					break
				# end if
			# end for
		# end for

		mp_dist = self.means_points_distance_matrix()
	
		# generate bound matrix for each point
		for i in range(self.data_size):
			# setting lower bound
			lower_bound = list()
			for k in range(self.cluster_num):
				lower_bound.append(mp_dist[i][k])
			# end for
			
			# setting upper bound
			upper_bound = min(lower_bound)
			
			# setting owner
			self.bound_matrix.append([lower_bound, upper_bound, lower_bound.index(upper_bound)])
		# end for


		for iter in range(self.max_iter) :

			# generate mean distance matrix
			mm_closeest_dist, mm_dist = self.means_means_distance_matrix()

			# pick mini_batch_size samples from data_set
			mini_batch = list()
			while len(mini_batch) < mini_batch_size :
				choose_num = int(self.data_size*np.random.random_sample())
				if not (choose_num in mini_batch) :
					mini_batch.append(choose_num)
				# end if
			# end while

			# center reassign
			for i in mini_batch:
				# if u(x) > s(c(x))
				if self.bound_matrix[i][1] > mm_closeest_dist[self.bound_matrix[i][2]] :
					# for each c != c(x), checking (1) u(x) > l(x,c) (2) u(x) > 0.5*d(c(x),c)
					for k in range(self.cluster_num):
						if k == self.bound_matrix[i][2] :
							continue
						# end if
						if self.bound_matrix[i][1] > self.bound_matrix[i][0][k] and \
					   	   self.bound_matrix[i][1] > mm_dist[self.bound_matrix[i][2]][k]:
							diff1 = points[i] - means[k]
							diff2 = points[i] - means[self.bound_matrix[i][2]]
							pk_dist = np.dot(diff1, diff1)
							pc_dist = np.dot(diff2, diff2)
							if pk_dist < pc_dist:
								self.bound_matrix[i][2]    = k
								self.bound_matrix[i][0][k] = pk_dist
								self.bound_matrix[i][1] 	  = pc_dist
							# end if
						# end if
					# end for		
				# end if
			# end for

			# update means
			v = [1 for i in range(self.cluster_num)]
			for i in mini_batch :
				centroid_index = self.bound_matrix[i][2]
				v[centroid_index] += 1
				step_size = 1/v[centroid_index]
				c = self.means[centroid_index]
				self.means[centroid_index] = c - np.multiply(step_size, c-self.data_set[i])
			# end for

			# calculate objection value
			obj = self.object_value_calc()
			if self.min_obj > obj :
				self.min_obj   = obj
				self.min_means = self.means
			# end if
			self.obj_record.append(obj)

			if iter < window_size :
				window_obj.append(obj)
			else :
				min_window_obj = min(window_obj)
				if min_window_obj*0.9 <= obj and obj < min_window_obj:
					print("iter: ",iter)
					return
				else :
					window_obj.append(obj)
					window_obj.popleft()
			# end if
		# end for

	# ============================== end ============================== #	

# ============================================= end ============================================= #	

def GenerateClusterDataSet(size, dimension, cluster_num, dist) :
	#np.random.seed(rnd_number)
	cluster_size = int(size/cluster_num)
	np.random.seed(0)
	w = np.random.randint(100, size=(1, 5)).flatten()
	C = np.random.normal(0, 1, (dimension, dimension))
	#for i in range(cluster_num) :
	initial = np.dot(np.random.randn(cluster_size, dimension), C)
	X = np.array(initial)
	for i in range(1, cluster_num) :
		XX = initial + i*dist*np.ones(dimension)
		X = np.append(X,XX,axis=0)
	# end for

	return X

# ============================================= end ============================================= #	

def ShowScatter(data_set, means) :
	color = ["#FF0000", "#00FF00", "#0000FF", "#DB7093", "#EEE8AA", "#FFEFD5", "#DDA0DD", "#9ACD32", "#F5DEB3", "#008080"]
	fig = plt.figure(figsize=(15, 10))
	#print("=== ",data_set[0]," , ",data_set[1])
	plt.scatter(data_set[:,0], data_set[:,1])
	for i in range(len(means)) :
		plt.scatter(means[i][0], means[i][1], c="#000000", s=100)
	# end for
	plt.show()

# ============================================= end ============================================= #	

def ShowObjVersusIteration(obj_value) :
	fig = plt.figure(figsize=(15, 10))
	iter = [ i for i in range(1,len(obj_value)+1) ]
	plt.plot(iter, obj_value)
	plt.show()

# ============================================= end ============================================= #	

def ShowMinObjVersusBatchSize(batch_list, obj_list) :
	fig = plt.figure(figsize=(15, 10))
	#print(data[:,0])
	plt.plot(batch_list, obj_list)
	plt.show()

# ============================================= end ============================================= #	

max_iter        = 500
cluster_num     = 5
dist            = 10
data_size       = 100
dimension       = 2
mini_batch_size = 10
StandardScaler = preprocessing.StandardScaler()
X = GenerateClusterDataSet(data_size, dimension, cluster_num, dist)
X = StandardScaler.fit_transform(X)

'''

	simple test for mini-batch kmeans algorithm

'''
#start = time.time()
#my_mini_batch_kmeans = MiniBatchKMeans(X, max_iter, cluster_num, mini_batch_size)
#my_mini_batch_kmeans.run()
#final = time.time()
#print("Execution Time: ",final-start)
#ShowScatter(X, my_mini_batch_kmeans.min_means)

'''

	test for objection value versus batch size

'''
'''
obj_list = list()
batch_list = list()
for batch_size in [5*i for i in range(1,20)] :
	my_mini_batch_kmeans.run()
	batch_list.append(batch_size)
	obj_list.append(my_mini_batch_kmeans.min_obj)
# end for
#ShowMinObjVersusBatchSize(batch_list, obj_list)
'''

