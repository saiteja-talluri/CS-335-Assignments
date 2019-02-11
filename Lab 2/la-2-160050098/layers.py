import numpy as np

class FullyConnectedLayer:
	def __init__(self, in_nodes, out_nodes):
		# Method to initialize a Fully Connected Layer
		# Parameters
		# in_nodes - number of input nodes of this layer
		# out_nodes - number of output nodes of this layer
		self.in_nodes = in_nodes
		self.out_nodes = out_nodes
		# Stores the outgoing summation of weights * feautres 
		self.data = None

		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))	
		self.biases = np.random.normal(0,0.1, (1, out_nodes))
		###############################################
		# NOTE: You must NOT change the above code but you can add extra variables if necessary 

	def forwardpass(self, X):
		# print('Forward FC ',self.weights.shape)
		# Input
		# activations : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_nodes]
		# OUTPUT activation matrix		:[n X self.out_nodes]

		###############################################
		# TASK 1 - YOUR CODE HERE
		self.data = np.matmul(X, self.weights) + self.biases
		return sigmoid(self.data)
		###############################################
		
	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		del_out_curr = np.multiply(delta,derivative_sigmoid(self.data))
		new_delta = np.matmul(del_out_curr,self.weights.T)
		weight_updation = np.matmul(activation_prev.T,del_out_curr)
		bias_updation = np.sum(del_out_curr,axis=0).reshape(self.biases.shape)
		self.weights -= lr*weight_updation
		self.biases -= lr*bias_updation
		return new_delta
		###############################################

class ConvolutionLayer:
	def __init__(self, in_channels, filter_size, numfilters, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for convolution layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer
		# numfilters  - number of feature maps (denoting output depth)
		# stride	  - stride to used during convolution forward pass
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = numfilters
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

		# Stores the outgoing summation of weights * feautres 
		self.data = None
		
		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))	
		self.biases = np.random.normal(0,0.1,self.out_depth)
		

	def forwardpass(self, X):
		# print('Forward CN ',self.weights.shape)
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE
		out = np.zeros([n,self.out_depth,self.out_row,self.out_col],dtype=float)
		self.data = out
		for i in range(0,n):
			for j in range(0,self.out_depth):
				for k in range(0,self.in_depth):
					out[i][j] += conv(X[i][k],self.weights[j][k],self.stride)
				out[i][j] += self.biases[j]
		self.data = out
		out = sigmoid(out)
		return out
		###############################################

	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		delta_out_curr = np.multiply(delta,derivative_sigmoid(self.data))
		new_delta = np.zeros_like(activation_prev)
		weight_updation = np.zeros_like(self.weights)
		bias_updation = np.zeros_like(self.biases)
		for u in range(0,n):
			for v in range(0,self.out_depth):
				for i in range(0,self.out_row):
					for j in range(0,self.out_col):
						weight_updation[v] += delta_out_curr[u][v][i][j]*activation_prev[u,:,i*self.stride : i*self.stride + self.filter_row,j*self.stride : j*self.stride + self.filter_col]
						bias_updation[v] += delta_out_curr[u][v][i][j]
						new_delta[u,:,i*self.stride : i*self.stride + self.filter_row,j*self.stride : j*self.stride + self.filter_col] += delta_out_curr[u][v][i][j]*self.weights[v]
		self.weights -= lr*weight_updation
		self.biases -= lr*bias_updation
		return new_delta
		###############################################
	
class AvgPoolingLayer:
	def __init__(self, in_channels, filter_size, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for max_pooling layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer

		# NOTE: Here we assume filter_size = stride
		# And we will ensure self.filter_size[0] = self.filter_size[1]
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = self.in_depth
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

	def forwardpass(self, X):
		# print('Forward MP ')
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE
		out = np.zeros([n,self.out_depth,self.out_row,self.out_col],dtype=float)
		for i in range(0,n):
			for j in range(0,self.in_depth):
					out[i][j] = avg(X[i][j],self.filter_row,self.filter_col,self.stride)
		return out
		###############################################


	def backwardpass(self, alpha, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# activations_curr : Activations of current layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		new_delta = np.zeros_like(activation_prev)
		delta_curr = delta/(self.filter_row*self.filter_col)
		for u in range(0,n):
			for v in range(self.out_depth):
				for i in range(0,self.out_row):
					for j in range(0,self.out_col):
						new_delta[u,v,i*self.stride : i*self.stride + self.filter_row,j*self.stride : j*self.stride + self.filter_col] = delta_curr[u,v,i,j]
		return new_delta
		###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        self.in_batch, self.r, self.c, self.k = X.shape
        return X.reshape(self.in_batch, self.r * self.c * self.k)

    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(self.in_batch, self.r, self.c, self.k)


# Helper Function for the activation and its derivative
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))

def conv(x,y,stride):
	(a,b) = x.shape
	(m,n) = y.shape
	out = np.zeros([int((a - m)/stride + 1),int((b - n)/stride + 1)],dtype=float)

	for i in range(0,a-m+1,stride):
		for j in range(0,b-n+1,stride):
			out[int(i/stride)][int(j/stride)] = np.sum(np.multiply(x[i:i+m,j:j+n],y))
	return out

def avg(x,m,n,stride):
	(a,b) = x.shape
	out = np.zeros([int((a - m)/stride + 1),int((b - n)/stride + 1)],dtype=float)

	for i in range(0,a-m+1,stride):
		for j in range(0,b-n+1,stride):
			out[int(i/stride)][int(j/stride)] = np.mean(x[i:i+m,j:j+n])
	return out