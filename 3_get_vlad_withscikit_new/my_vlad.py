from sklearn.utils.extmath import squared_norm
import os, sys
import image_norm_test as myData;
import numpy as np
import time
from sklearn.utils.extmath import safe_sparse_dot

from sklearn.metrics.pairwise import euclidean_distances


class my_vlad(object):

    def __init__(self , centroids):
        self.unitVec = []
        for item in centroids:
    #        print(item)
            item_norm = np.linalg.norm(item)
    #        print(item/item_norm)
            self.unitVec.append(item/item_norm)
        
    def get_vlad(self,datas) :
        result = []
        ### Assign data to nearest centriods

	start = time.time()

        mapping = self.NN(datas,self.unitVec)
	finish = time.time()
	print("Execution Time: ",finish-start," sec")

        #print("mapping: ",mapping)
        
        start = time.time()

	for item in mapping:
            summ = np.sum(np.array(item),0)
            if(summ.all() != 0):
                ### used intra_normalization ###   
                ans = self.normalization(summ)
            else :
                ans = np.zeros(len(self.unitVec[0]))
            result.append(ans.tolist())
#        print(result)
        result = self.normalization(result)
	finish = time.time()
        print("Execution Time: ",finish-start," sec")


#        print(result.tolist())
        #return sum( cosine_similarity(data,centroids))
        return result
    
    def normalization(self,summ):
        
        return (summ/np.linalg.norm(summ))
    
    def NN(self,datas, centroids):
      #  start = time.time()
        ####### find which centroids the x is closet to, and put x into the centroids location #####
        group = [[] for n in range(len(centroids))]
#        group = [[np.zeros(len(centroids[0]))] for n in range(len(centroids))]

        all_distances = euclidean_distances(centroids, datas, squared=True)

	labels = np.empty(len(datas), dtype=np.int32)
        labels.fill(-1)
        mindist = np.empty(len(datas))
        mindist.fill(np.infty)
        for center_id in range(len(centroids)):
            dist = all_distances[center_id]
            labels[dist < mindist] = center_id
            mindist = np.minimum(dist, mindist)
        #for k in range(len(centroids)):
	    #group.append(list)
	for i in range(len(labels)):
	    group[labels[i]].append(datas[i]-centroids[labels[i]])


        # end for
#        print("End of NN: ",(time.time()-start))
        return group

    
