from sklearn.utils.extmath import squared_norm
import os, sys
import image_norm_test as myData;
import numpy as np
import time
from sklearn.utils.extmath import safe_sparse_dot




A = [ [1,1] , [1,1] , [1,1] ]
A = np.asarray(A)
B = safe_sparse_dot(A,A.T)
print(B)
