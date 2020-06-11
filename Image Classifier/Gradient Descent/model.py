from load_data import load_data
from matplotlib import pyplot as plt
import numpy as np
from dnn_utils import relU, sigmoid, relU_back, sigmoid_back

# dataset preparation
def data_prep(file_name):
	X_train_orig, Y_train, X_test_orig, Y_test = load_data(file_name)
	m_train = X_train_orig.shape[0]
	m_test = X_test_orig.shape[0]

	X_train_flatten = X_train_orig.reshape(m_train,-1).T
	X_test_flatten = X_test_orig.reshape(m_test,-1).T
	X_train = X_train_flatten/255
	X_test = X_test_flatten/255		

	mean_train = np.mean(X_train,axis=1,keepdims=True)
	mean_test = np.mean(X_test,axis=1,keepdims=True)
	std_train = np.std(X_train,axis=1,keepdims=True)
	std_test = np.std(X_test,axis=1,keepdims=True)


	X_train = (X_train - mean_train)/std_train
	X_test = (X_test - mean_test)/std_test

	return X_train, Y_train, X_test, Y_test

def initialize_parameters(layers_dims):
	parameters = {}
	n_L = len(layers_dims)-1	
	for i in range(1, n_L+1):
		parameters['W'+str(i)] = np.random.randn(layers_dims[i],layers_dims[i-1]) * 0.01
		parameters['b'+str(i)] = np.zeros((layers_dims[i],1))	
		# print('W'+str(i)+':',parameters['W'+str(i)], 'b'+str(i)+':', parameters['b'+str(i)])	

	return parameters

def forward_prop(X, parameters):

	caches = {}
	n_L =  len(parameters) // 2	
	caches['A0'] = X
	for i in range(1, n_L):
		caches['Z'+str(i)] = np.dot(parameters['W'+str(i)],caches['A'+str(i-1)]) + parameters['b'+str(i)]
		caches['A'+str(i)] = relU(caches['Z'+str(i)])
	# print('WL:',parameters['W'+str(n_L)])
	# print('bL:',parameters['b'+str(n_L)])
	# print('AL-1:',caches['A'+str(n_L-1)])
	caches['Z'+str(n_L)] = np.dot(parameters['W'+str(n_L)],caches['A'+str(n_L-1)]) + parameters['b'+str(n_L)]
	# print('ZL:',caches['Z'+str(n_L)])
	caches['A'+str(n_L)] = sigmoid(caches['Z'+str(n_L)])		
	# print('AL:',caches['A'+str(n_L)])
	return caches

def compute_cost(Y,AL):	
	m = Y.shape[1]		
	# print(AL)
	cost = -(1/m) * np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))
	cost = np.squeeze(cost)		
	return cost

def backward_prop(caches, parameters, Y, AL):
	grads = {}
	n_L = len(parameters)//2	
	m = Y.shape[1]	
	grads['dA'+str(n_L)] = (1/m)*np.sum(-np.divide(Y,AL)+np.divide(1-Y,1-AL))	
	for i in reversed(range(1, n_L+1)):			
		grads['dZ'+str(i)] = grads['dA'+str(i)] * relU_back(caches['Z'+str(i)])
		grads['dW'+str(i)] = np.dot(grads['dZ'+str(i)], caches['A'+str(i-1)].T)
		grads['db'+str(i)] = np.sum(grads['dZ'+str(i)], axis=1, keepdims=True)
		grads['dA'+str(i-1)] = np.dot(parameters['W'+str(i)].T, grads['dZ'+str(i)])		
	return grads

def update_parameters(grads, parameters, learning_rate):
	n_L = len(parameters) // 2	
	# print('W:',parameters['W'+str(n_L)])
	# print('dW:',grads['dW'+str(n_L)])
	for i in range(1,n_L+1):
		parameters['W'+str(i)] = parameters['W'+str(i)] - learning_rate*grads['dW'+str(i)]
		parameters['b'+str(i)] = parameters['b'+str(i)] - learning_rate*grads['db'+str(i)] 

	return parameters

class model:
	parameters = []
	caches = []
	layers_dims = []
	n_L = 0
	learning_rate = 0

	def __init__(self, layers_dims):
		self.layers_dims = layers_dims
		self.n_L = len(layers_dims) - 1
		print('-------------model initialized----------')
		print('No. of layers: ',self.n_L)
		print('unit distribution: ',self.layers_dims)

	def train(self,X, Y, learning_rate = 0.001, epochs = 1000):				
		parameters = initialize_parameters(self.layers_dims)
		n_L = self.n_L
		self.learning_rate = learning_rate

		for i in range(1,epochs+1):
			caches = forward_prop(X,parameters)
			cost = compute_cost(Y, caches['A'+str(n_L)])
			grads = backward_prop(caches, parameters, Y, caches['A'+str(self.n_L)])				
			parameters = update_parameters(grads, parameters, learning_rate)			
			if(i%100 == 0):
				print('cost after', i, 'iterations: ', cost)
		self.parameters = parameters
		self.caches = caches
	
	def predict(self,X,Y):
		m = X.shape[1]
		caches = self.caches
		parameters = self.parameters
		n_L = self.n_L		
		caches = forward_prop(X,parameters)		
		print((caches['A'+str(n_L)]))
		pred = np.squeeze((caches['A'+str(n_L)]>0.5)*1)
		Y = np.squeeze(Y)				
		print(pred,Y)
		accuracy = (np.sum(pred == Y)/m)*100
		return accuracy


def main():
	np.random.seed(1)
	dataset_name = 'chest_xray'
	X_train, Y_train, X_test, Y_test = data_prep(dataset_name)
	layers_dims = [49152, 20, 7, 7, 7, 5, 1]
	cur_model = model(layers_dims)
	cur_model.train(X_train, Y_train)
	acc = cur_model.predict(X_test, Y_test)
	print('Accuracy:',acc)

if __name__ == "__main__":	
	main()