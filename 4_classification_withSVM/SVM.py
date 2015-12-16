
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import svm
from collections import deque
import numpy as np
import math
import matplotlib.pyplot as plt
import time

class MySVM :
	def __init__(self, alpha, beta, C, h, max_iter, step_size, rnd_number, train_X, train_y, validation_X, validation_y, \
			     gradient_error, improve_ratio, eta0=1, t0=1) :
		train_size, point_dim = train_X.shape
		self.alpha            = alpha
		self.beta             = beta
		self.C                = C
		self.h     	          = h
		self.max_iter         = max_iter
		self.step_size        = step_size
		self.rng              = np.random.RandomState(rnd_number)
		self.train_X          = train_X
		self.train_y          = train_y
		self.validation_X     = validation_X
		self.validation_y     = validation_y
		self.gradient_error   = gradient_error
		self.improve_ratio    = improve_ratio
		self.eta0			  = eta0
		self.t0 			  = t0
		self.method           = 1
		self.point_dim        = point_dim
		self.train_size       = train_size
		self.validation_size  = validation_X.shape[0]
		self.w                = np.zeros(self.point_dim)
		#self.w               = np.ones(self.point_dim)
		self.loss_c           = 1/(4*h)
		self.grad_loss_c      = 1/(2*h)
		self.loss_c1          = 1+h
		self.loss_c2          = 1-h
		self.obj_record       = list()

	# ============================== end ============================== #	

	def grad_checker() :
		yt   = self.validate(self.train_y, np.dot(self.w, np.transpose(self.train_X)), self.train_size)
		loss = self.loss_calc(yt)
		self.compute_obj(loss, self.train_size)


	# ============================== end ============================== #

	# calculate w(d X 1) dot x(d X n) , dimension n
	def predict(self, x, w = False) :
		if not w :
			w = self.w
		# end if 
		return np.dot(w, np.transpose(x))

	# ============================== end ============================== #	

	# calculate y(n X n) dot wx (n X 1) , dimension n 
	def validate(self, y, wx, size) :
		yt = list()
		for i in range(len(y)) :
			yt.append(y[i]*wx[i])
		# end for
		return yt

	# ============================== end ============================== #	

	def loss_calc(self, yt):
		loss_list = list()

		for item in yt :
			if item > self.loss_c1:
				loss_list.append(0.0)
			elif item < self.loss_c2:
				loss_list.append(1-item)
			else:
				loss_list.append(math.pow(self.loss_c1-item, 2)*self.loss_c)
			# end if
		# end for

		return loss_list

	# ============================== end ============================== #	

	def grad_loss_calc(self, yt, x, y):
		grad_loss_list = list()

		for i in range(len(yt)) :
			if yt[i] > self.loss_c1:
				grad_loss_list.append(np.zeros(self.point_dim))
			elif yt[i] < self.loss_c2:
				grad_loss_list.append(-y[i]*x[i])
			else:
				grad_loss_list.append((self.loss_c1-yt[i])*self.grad_loss_c*(-y[i]*x[i]))
			# end if
		# end for
		return np.transpose(grad_loss_list)

	# ============================== end ============================== #	

	# sum of n loss value 
	# grad_loss is d X n 
	# grad_loss_sum is d X 1
	def grad_loss_sum_calc(self, grad_loss) :
		grad_loss_sum = list()
		for item in grad_loss:
			grad_loss_sum.append(np.sum(item))
		# end for
		return grad_loss_sum


	# ============================== end ============================== #	

	def compute_obj(self, loss, size, w = []) :
		if w == [] :
			w = self.w
		# end if
		return np.dot(np.transpose(w), w) + self.C/size*np.sum(loss)

	# ============================== end ============================== #	

	def compute_grad(self, grad_loss, size, w = []) :
		if w == [] :
			w = self.w
		# end if
		grad_loss_sum = self.grad_loss_sum_calc(grad_loss)
		return np.multiply(2.0, w) + np.multiply(self.C/size, grad_loss_sum)

	# ============================== end ============================== #	

	def validation_check(self, w) :
		error = 0
		predict_list = np.dot(w, np.transpose(self.validation_X))
		for i in range(len(self.validation_y)) :
			if  predict_list[i]*self.validation_y[i] < 0 :
				error += 1
			# end if
		# end for
		return error/self.validation_size

	# ============================== end ============================== #	

	
	def backtracking_line_search(self, current_obj_val, step_size, grad_loss, x, y, size) :
		#grad_loss_norm_square = norm_wo_root(grad_loss)
		grad_loss_norm_square = np.dot(grad_loss, grad_loss)
		c1 = self.alpha*grad_loss_norm_square
		while step_size > 1e-5 :
			new_w       = self.w - np.multiply(step_size, grad_loss)
			new_yt      = self.validate(np.dot(new_w, np.transpose(x)), y, size)
			new_loss    = self.loss_calc(new_yt)
			new_obj_val = self.compute_obj(new_loss, size, new_w)
			if (new_obj_val - (current_obj_val - c1*step_size)) < 1e-6 :
				return step_size
			# end if
			step_size = self.beta*step_size
		# end while
		return step_size
	
	# ============================== end ============================== #

	def my_gradient_decent(self, option=3, fix_size=False) :
		self.w = np.zeros(self.point_dim)
		self.obj_record = list()
		perfect_misclassify_error = 1.0
		perfect_w = np.zeros(self.point_dim)
		step_size = self.step_size
		window_size = 10
		error_window = deque()

		#perfect_time
		#start = time.time()
		for iter in range(self.max_iter) :
			count = 0
			yt              = self.validate(self.train_y, np.dot(self.w, np.transpose(self.train_X)), self.train_size)
			loss            = self.loss_calc(yt)
			grad_loss       = self.grad_loss_calc(yt, self.train_X, self.train_y)
			grad_val        = self.compute_grad(grad_loss, self.train_size)
			current_obj_val = self.compute_obj(loss, self.train_size)
			
			if not fix_size : 
				step_size = self.backtracking_line_search(current_obj_val, step_size, grad_val, self.train_X, self.train_y, self.train_size)
			# end if

			# update weight
			#print(self.w," , ",np.multiply(step_size, grad_val))
			old_w = self.w
			self.w = self.w - np.multiply(step_size, grad_val)
			self.obj_record.append(current_obj_val)

			if option == 2 :
				if np.linalg.norm(grad_val) < self.gradient_error :
					if np.linalg.norm(self.w) != 0 :
						self.w = self.w/np.linalg.norm(self.w)
					return
				# end if
			elif option == 3 :
				current_misclassify_error = self.validation_check(self.w)
				if current_misclassify_error < perfect_misclassify_error :
					perfect_misclassify_error = current_misclassify_error
					perfect_w = self.w
				# end if
				if len(error_window) < window_size :
					error_window.append(current_misclassify_error)
				else :
					window_min = min(error_window)		
					if current_misclassify_error == 0 or ((current_misclassify_error <= window_min) and \
						((current_misclassify_error/window_min) > self.improve_ratio))  :
						self.w = perfect_w
						if np.linalg.norm(self.w) != 0 :
							self.w = self.w/np.linalg.norm(self.w)
						return 
					else:
						error_window.append(current_misclassify_error)
						error_window.popleft()	
					# end if
				# end if
			# end if
		# end for

		if option == 3 :
			self.w = perfect_w
		# end if
		if np.linalg.norm(self.w) != 0 :
			self.w = self.w/np.linalg.norm(self.w)


	# ============================== end ============================== #

	# step_size_mode = 0 (fixed size)
	# step_size_mode = 1 (decrease by iteration)
	# step_size_mode = 2 (backtracking line search)
	def my_sgd(self, option=3, step_size_mode=0) :
		self.w = np.zeros(self.point_dim)
		self.obj_record = list()
		perfect_misclassify_error = 1.0
		perfect_w = list()
		window_size = int(self.train_size/2)
		#window_size = 50
		error_window = deque()
		step_size = self.step_size
		self.obj_record = list()

		for iter in range(1, self.max_iter+1) :
			
			self.rng    = np.random.RandomState(iter)
			permutation = self.rng.permutation(len(self.train_X))
			self.train_X, self.train_y = self.train_X[permutation], self.train_y[permutation]

			if step_size_mode == 1:
				step_size = self.eta0/(self.t0 + iter)
			# end if

			obj_sum = 0
			for i in range(self.train_size) :
				yt              = self.train_y[i]*np.dot(self.w, np.transpose(self.train_X[i]))
				loss            = self.loss_calc([yt])
				grad_loss       = self.grad_loss_calc([yt], [self.train_X[i]], [self.train_y[i]])
				grad_val        = self.compute_grad(grad_loss, 1)
				current_obj_val = self.compute_obj(loss, 1)
				obj_sum        += current_obj_val

				if step_size_mode == 2 : 
					step_size = self.backtracking_line_search(current_obj_val, step_size, grad_val, [self.train_X[i]], [self.train_y[i]], 1)
				# end if

				# update weight
				self.w = self.w - np.multiply(step_size, grad_val)
	
				if option == 3 :
					current_misclassify_error = self.validation_check(self.w)
					if current_misclassify_error < perfect_misclassify_error :
						perfect_misclassify_error = current_misclassify_error
						perfect_w = self.w
					# end if
					if len(error_window) < window_size :
						error_window.append(current_misclassify_error)
					else :
						window_min = min(error_window)
						if current_misclassify_error == 0 or ((current_misclassify_error <= window_min) and \
							((current_misclassify_error/window_min) > self.improve_ratio))  :
							self.w = perfect_w
							return 
						else:
							error_window.append(current_misclassify_error)
							error_window.popleft()	
						# end if
					# end if
				# end if
			# end for
			self.obj_record.append(obj_sum/self.train_size)
		# end for
		if option == 3 :
			self.w = perfect_w
		# end if
		self.w = self.w/np.linalg.norm(self.w)

	# ============================== end ============================== #	

	def fit(self, train_X, train_y) :
		# method = 1 (default value for gradient decent method)
		# mehtod = 2 (SGD method)

		self.train_X = train_X
		self.train_y = train_y
		if self.method == 1 :
			self.my_gradient_decent()
		elif self.method == 2 :
			self.my_sgd()
		# end if

	# ============================== end ============================== #
