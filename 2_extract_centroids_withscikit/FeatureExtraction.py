'''

Homework 4

Name     : Tailin Lo
NetID    : tl1720
N number : N15116873
Email    : tl1720@nyu.edu

======================================

FeatureExtraction Lib 

'''

#import KMeans
import FetchFile
#import MiniBatchKMeans
import time
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
# ============================================= end ============================================= #	

def cos_similarity(v1, v2, dim) :
	v1_norm   = 0.0
	v2_norm   = 0.0
	v1_dot_v2 = 0.0
	for d in range(0, dim):
		v1_dot_v2 += v1[d]*v2[d]
		v1_norm   += v1[d]*v1[d]
		v2_norm   += v2[d]*v2[d]
	# end for
	return v1_dot_v2/(math.sqrt(v1_norm*v2_norm))


def cos_similarity_speedup(v1, v2, dim) :
	v1_dot_v2 = 0.0
	for d in range(0, dim):
		v1_dot_v2 += v1[d]*v2[d]
	# end for
	return v1_dot_v2

	
# ============================================= end ============================================= #	

def feature_project(features, point, feature_dim, point_dim) :
	feature_vector = list()
	for d in range(0, feature_dim):
		feature_vector.append(cos_similarity(features[d], point, point_dim))
	# end for
	return feature_vector


# ============================================= end ============================================= #	


def learnvocabulary(train_set, cluster_num, max_iter, rnd_number) :
	start = time.time()
	print("Start Kmeans")
	my_mini_batch_kmeans = MiniBatchKMeans(n_clusters = cluster_num, random_state = rnd_number , init_size=3000).fit(train_set)
	
	means = my_mini_batch_kmeans.cluster_centers_ 
	print(means)
	print("Kmeans Time For Cluster Number %d: " %cluster_num ,time.time()-start)
	return means

# ============================================= end ============================================= #	

def getbof(features, point, feature_dim, point_dim):
	feature_vector = list()
	for d in range(0, feature_dim):
		feature_vector.append(cos_similarity(features[d], point, point_dim))
	# end for
	return feature_vector


def getbof_speedup(features, point, feature_dim, point_dim):
	feature_vector = list()
	for d in range(0, feature_dim):
		feature_vector.append(cos_similarity_speedup(features[d], point, point_dim))
	# end for
	return feature_vector

# ============================================= end ============================================= #	


def trainset_project(features, trainset, speedup=False) :
	feature_vectors = list()
	feature_dim 	= len(features)
	point_dim   	= len(trainset[0])
	start = time.time()
	if not speedup:
		for point in trainset:
			feature_vectors.append(getbof(features, point, feature_dim, point_dim))
		# end for
	else:
		for point in trainset:
			feature_vectors.append(getbof_speedup(features, point, feature_dim, point_dim))
		# end for
	print("Projection Time: ",time.time()-start)
	start = time.time()
	transformer = TfidfTransformer()
	feature_vectors_tfidf = transformer.fit_transform(feature_vectors)
	print("TFIDF Transsform Time: ",time.time()-start)
	return feature_vectors


# ============================================= end ============================================= #	

def mynormalize(v):
	v_norm = list()
	norm = np.linalg.norm(v)
	for v_i in v:
		v_norm.append(v_i/norm)
	# end for
	return v_norm


def mynormalize_multi(v_multi):
	v_multi_norm = list()
	for v in v_multi:
		v_multi_norm.append(mynormalize(v))
	# end for
	return v_multi_norm


# ============================================= end ============================================= #	

def write_features(file_name, features):
	f = open(file_name, 'w+')
	for feature in features:
		for d in feature:
			f.writelines(str(d)+" ")
		# end for
		f.writelines("\n")
	# end for
	f.close()


# ============================================= end ============================================= #	

def read_features(file_name):
	features = list()

	f = open(file_name)
	for line in f.readlines():
		feature_list = line.split(" ")
		feature = list()
		for item in feature_list:
			if item != '\n':
				feature.append(float(item))
			# end if
		# end for
		features.append(feature)
	# end for
	f.close()
	return features


# ============================================= end ============================================= #	

def gen_feature_fname(class_num, subsample_size, window_size, cluster_num):
	return str(class_num)+"_"+str(subsample_size)+"_"+str(window_size)+"_"+str(cluster_num)+".txt"


# ============================================= end ============================================= #	

def extract_feature_by_kmeans(class_num, subsample_size, window_size, cluster_num, max_iter, rnd_number) :
	file_name = gen_feature_fname(class_num, subsample_size, window_size, cluster_num)
	print(file_name)
	start = time.time()
	train_X, test_X, train_y, test_y = FetchFile.gen_data(class_num, subsample_size, window_size, rnd_number)
	print("Generate Data Time: ",time.time()-start)
	features = learnvocabulary(train_X, cluster_num, max_iter, rnd_number)
	features = mynormalize_multi(features)
#	print(file_name , len(features))
	write_features(file_name, features)
	print("=== Feature Extraction Finish ===")

# ============================================= end ============================================= #	

if __name__ == "__main__":
	
	'''

		Unit Test for extract_feature_by_kmeans
		By uncomment the following code, you can run a simple feature extraction case (1 class/1 image/3 clusters)

	'''
	'''
	rnd_number 	   = 8131985
	class_num 	   = 1
	subsample_size = 1
	window_size    = 3
	cluster_num    = 3
	max_iter 	   = 50
	extract_feature_by_kmeans(class_num, subsample_size, window_size, cluster_num, max_iter, rnd_number)
	'''


