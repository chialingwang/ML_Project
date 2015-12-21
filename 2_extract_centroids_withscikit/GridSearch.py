'''

Homework 4

Name     : Tailin Lo
NetID    : tl1720
N number : N15116873
Email    : tl1720@nyu.edu

'''

#import FetchFile
import FeatureExtraction
#import Classification
#import multiprocessing


def grid_search_for_neighbor(class_num, subsample_size, window_size, cluster_num, max_iter, rnd_number, neighbor_num_seq):
	train_X, test_X, train_y, test_y = FetchFile.gen_data(class_num, subsample_size, window_size, rnd_number)
	#for neighbor_num in [2**i for i in range(neighbor_log2_num)]:
	for neighbor_num in neighbor_num_seq:
		Classification.classifiy(class_num, subsample_size, window_size, cluster_num, max_iter, rnd_number, neighbor_num, train_X, train_y, test_X, test_y)
	# end for

# ============================================= end ============================================= #	


def grid_search_for_neighbor_multiprocess(class_num, subsample_size, window_size, cluster_num, max_iter, rnd_number, neighbor_num_seq):
	#Classification.classifiy(class_num, subsample_size, window_size, cluster_num, max_iter, rnd_number, neighbor_num)
	train_X, test_X, train_y, test_y = FetchFile.gen_data(class_num, subsample_size, window_size, rnd_number)
	jobs = []
	#for neighbor_num in [2**i for i in range(neighbor_log2_num)]:
	for neighbor_num in neighbor_num_seq:
		p = multiprocessing.Process(target=Classification.classifiy, args=(class_num, subsample_size, window_size, cluster_num, \
			max_iter, rnd_number, neighbor_num, train_X, train_y, test_X, test_y))
		jobs.append(p)
		p.start()
	# end for
	print(jobs)

# ============================================= end ============================================= #	


def grid_search_for_cluster(class_num, subsample_size, window_size, max_iter, rnd_number, cluster_num_seq):
	for cluster_num in cluster_num_seq:
		FeatureExtraction.extract_feature_by_kmeans(class_num, subsample_size, window_size, cluster_num, max_iter, rnd_number)
	# end for

# ============================================= end ============================================= #	

rnd_number 	   = 8131985
class_num 	   = __CLASS_NUM__
#class_num         = 10
subsample_size = 92
window_size    = [__WINDOW__]
max_iter 	   = 50

'''

	Grid Search by sweeping cluster number and neighbor number
	Users can use grid_search_for_neighbor_multiprocess function to parallelly execute jobs (not larger than your CPU size - 1)
	Note: 
	1. grid_search_for_cluster only generates centroids file, and users have to use grid_search_for_neighbor to find the optimal value.
	2. a simple unit test by setting class_num = 1, subsample_size = 1, cluster_num = 3
	3. before executing a simple unit test, ensuring there are a corresponding centroids file

'''

# [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] or use [2**i for i in range(0,11)]
#grid_search_for_cluster(class_num, subsample_size, window_size, max_iter, rnd_number, [16] )
for wsize in window_size:
	grid_search_for_cluster(class_num, subsample_size, wsize, max_iter, rnd_number, [__CLUSTER_NUM__])
#       grid_search_for_cluster(class_num, subsample_size, wsize, max_iter, rnd_number, [4])

#grid_search_for_neighbor(class_num, subsample_size, window_size, cluster_num, max_iter, rnd_number, [16, 32, 64, 128, 256])

