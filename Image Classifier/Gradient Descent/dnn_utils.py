import numpy as np

def relU(Z):
	A = np.maximum(0.01*Z, Z)
	return A

def relU_back(Z):	
	dZ = Z>0	
	dZ = dZ*1	
	return dZ

def sigmoid(Z):
	A = 1/(1+np.exp(-Z))
	return A

def sigmoid_back(Z):
	dZ = sigmoid(Z) * (1-sigmoid(Z))
	return dZ