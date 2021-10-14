
import numpy as np

class Perceptron(object):

	#percepton constructor, input number iterations and learning rate
	def __init__(self, inputAmount, epochs = 20, learning_rate = 0.01):
	
		self.epochs = epochs	        
		self.learning_rate = learning_rate
		self.weights = np.zeros (inputAmount + 1)

	#using precalculated weights predict what the inputs should give
	def predict(self, inputs):
		
		inputs = inputs.astype(float)		
		acca = np.dot(inputs, self.weights[1:]) + self.weights[0]
		
		if acca > 0:		
			binOut = 1
			
		else:			
			binOut = 0
			
		return binOut

	#training function to achieve desired weights
	def train (self, training_inputs, expected_outputs):   
	
		
		for _ in range (self.epochs):
					
			for inputs, expected in zip (training_inputs, expected_outputs):
				
				inputs = inputs.astype(float)		
				acca = np.dot(inputs, self.weights[1:]) + self.weights[0]
				if acca > 0:
					binOut = 1
				else:
					binOut = 0				

				self.weights[1:] += self.learning_rate * (expected - binOut) * inputs
				self.weights[0] += self.learning_rate * (expected - binOut)

