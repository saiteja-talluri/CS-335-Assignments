import numpy as np
from utils import *

def preprocess(X, Y):
	''' TASK 0
	X = input feature matrix [N X D] 
	Y = output values [N X 1]
	Convert data X, Y obtained from read_data() to a usable format by gradient descent function
	Return the processed X, Y that can be directly passed to grad_descent function
	NOTE: X has first column denote index of data point. Ignore that column 
	and add constant 1 instead (for bias part of feature set)
	'''

	N,D = X.shape
	no_cols = 1

	for i in range(1,D):
		if(isinstance(X[0][i], str)):
			no_cols += len(set(X[:,i]))
		no_cols += 1

	X_new = np.ones((N,no_cols), dtype= 'float')
	Y_new = Y.astype(float)

	col = 1
	for i in range(1,D):
		if(isinstance(X[0][i], str)):
			X_new[:,col:col+len(set(X[:,i]))] = one_hot_encode(X[:,i], set(X[:,i]))
			col += len(set(X[:,i]))
		else:
			mean = np.mean(X[:,i])
			sd = np.std(X[:,i])
			X_new[:,col] = (X[:,i]-mean)/sd
			col += 1
	
	return X_new,Y_new

def grad_ridge(W, X, Y, _lambda):
	'''  TASK 2
	W = weight vector [D X 1]
	X = input feature matrix [N X D]
	Y = output values [N X 1]
	_lambda = scalar parameter lambda
	Return the gradient of ridge objective function (||Y - X W||^2  + lambda*||w||^2 )
	'''

	gradient = ((-2)*np.dot(X.transpose(), Y - np.dot(X,W))) + 2*_lambda*W
	return gradient

def ridge_grad_descent(X, Y, _lambda, max_iter=30000, lr=0.00001, epsilon = 1e-2):
	''' TASK 2
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	lr 			= learning rate
	epsilon 	= gradient norm below which we can say that the algorithm has converged 
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	NOTE: You may precompure some values to make computation faster
	'''
	W = np.random.normal(0,0.01,(X.shape[1], 1))

	for i in range(0, max_iter):
		gradient = grad_ridge(W, X, Y, _lambda);
		if(np.linalg.norm(gradient, ord=2) < epsilon):
			W = W - lr*gradient
			break
		else:
			W = W - lr*gradient
	return W

def k_fold_cross_validation(X, Y, k, lambdas, algo):
	''' TASK 3
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	k 			= number of splits to perform while doing kfold cross validation
	lambdas 	= list of scalar parameter lambda
	algo 		= one of {coord_grad_descent, ridge_grad_descent}
	Return a list of average SSE values (on validation set) across various datasets obtained from k equal splits in X, Y 
	on each of the lambdas given 
	'''
	N,D = X.shape
	frac = float(N)/float(k)

	ans = []
	for _lambda in lambdas:
		sse_list = []
		for i in range(k):
			l_bound, r_bound = int(i*frac), int((i+1)*frac)
			X_train = np.zeros((N -r_bound + l_bound, D))
			Y_train = np.zeros((N -r_bound + l_bound, 1))

			X_train[0:l_bound, :] = X[0:l_bound, :]
			X_train[l_bound:, :] = X[r_bound:, :]
			Y_train[0:l_bound, :] = Y[0:l_bound, :]
			Y_train[l_bound:, :] = Y[r_bound:, :]
			X_test = X[l_bound:r_bound, :]
			Y_test = Y[l_bound:r_bound, :]

			W_trained = algo(X_train, Y_train, _lambda)
			sse_list.append(sse(X_test, Y_test, W_trained))
		ans.append(np.mean(sse_list))
		print("Lambda : " + str(_lambda) + ", Ans : " + str(ans[-1]))
	return ans

def coord_grad_descent(X, Y, _lambda, max_iter=1000):
	''' TASK 4
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	'''
	N,D = X.shape
	W = np.random.normal(0,0.01,(D, 1))

	preprocess = np.zeros(D)
	for i in range(D):
		preprocess[i] = np.dot(X[:,i].T,X[:,i])

	for i in range(0, max_iter):
		not_changed = 0
		for j in range(D):
			if preprocess[j] == 0:
				W[j] = 0
			else:
				rho_j = (np.dot(X[:,j].T,Y - np.dot(X,W))) + preprocess[j]*W[j]
				if(rho_j < (-0.5)*_lambda):
					beta_j = (rho_j + (0.5)*_lambda)/preprocess[j]
					if(W[j] == beta_j):
						not_changed += 1
					else:
						W[j] = beta_j
				elif (rho_j > (0.5)*_lambda):
					beta_j = (rho_j - (0.5)*_lambda)/preprocess[j]
					if(W[j] == beta_j):
						not_changed += 1
					else:
						W[j] = beta_j
				else:
					if(W[j] == 0):
						not_changed += 1
					else:
						W[j] = 0
		if not_changed == D:
			break
	return W

if __name__ == "__main__":
	# Do your testing for Kfold Cross Validation in by experimenting with the code below 
	X, Y = read_data("./dataset/train.csv")
	X, Y = preprocess(X, Y)
	trainX, trainY, testX, testY = separate_data(X, Y)

	lambdas = [...] # Assign a suitable list Task 5 need best SSE on test data so tune lambda accordingly
	scores = k_fold_cross_validation(trainX, trainY, 6, lambdas, coord_grad_descent)

	'''
	lambdas = [300000, 310000, 320000, 330000, 340000, 350000, 370000, 400000, 410000, 420000, 430000, 440000, 450000]
	scores =  [168839043350.3544, 168724745503.99258, 168652817057.73956, 168609570271.94696,168591968665.28323, 168610986799.54117, 168743850943.06015, 168837183139.23697, 168805133677.0105, 168789684756.9399, 168791631804.54233, 168811506871.57938, 168854360254.16794]
	'''

	# plot_kfold(lambdas, scores)




