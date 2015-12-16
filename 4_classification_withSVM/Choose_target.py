import numpy as np

class Choose_Target(object):

    def __init__(self , Xin , yin):
        self.X = Xin
        self.y = yin

    def select(self , i ):

        for index in range(len(self.y)):
                if(self.y[index] == i):
			#print("This is class %d" %i)
                        self.y[index] = -1
                else:
			#print("This is others")
                        self.y[index] = 1

