aClass = input("please enter your first class for classification (class-1, class-2, class-3) : ")
bClass = input("please now enter your second class to linearly seperate from the first (class-1, class-2, class-3) : ")

# ------- TRAINING PORTION

import numpy as np
from Perceptron import Perceptron

f = open('train.data')
trainData = []
np.array(trainData,dtype=float)
expected_outputs = []

#-------------- read in training data and split

for line in f:

	lines = line.rstrip()	
	linesplit = lines.split(',')
	
	if linesplit[4] == aClass:
	
		trainData.append(np.array([linesplit[0],linesplit[1],linesplit[2],linesplit[3]]))		
		expected_outputs.append(1)
		
	elif linesplit[4] == bClass:
	
		trainData.append(np.array([linesplit[0],linesplit[1],linesplit[2],linesplit[3]]))		
		expected_outputs.append(0)

f.close()
	
Perceptron = Perceptron(4)
Perceptron.train(trainData,expected_outputs)

# --------------- read in test data




f2 = open ('test.data')

testData = []
expected_test_outputs = []
np.array(testData, dtype = float)

for line in f2:

	lines = line.rstrip()
	linesplit = lines.split(',')
	
	if linesplit[4] == aClass:
	
		testData.append(np.array([linesplit[0],linesplit[1],linesplit[2],linesplit[3]]))		
		expected_test_outputs.append(linesplit[4])
		
	elif linesplit[4] == bClass:
	
		testData.append(np.array([linesplit[0],linesplit[1],linesplit[2],linesplit[3]]))		
		expected_test_outputs.append(linesplit[4])
		
f2.close()





#--------------- check accuracy of perceptron

# Use iterator j to track expected_test_outputs in our for loop
j = 0

# We increment correctPredictions if the machine gets it right
correctPredictions = 0

for i in testData:

	print ("Data input: ",i)	
	print ("Expected result: ",expected_test_outputs[j])
	print ("Perceptron output: ", Perceptron.predict(i))
	
	if (Perceptron.predict(i) == 1):
	
		if (expected_test_outputs[j] == aClass):
		
			correctPredictions += 1		
			print ("Actual result: ",aClass,"\n")
			
		else:
		
			print ("Actual result: ",bClass,"\n")
		
	else:
	
		if (expected_test_outputs[j] == bClass):
		
			correctPredictions += 1	
			print ("Actual result: ",bClass,"\n")
			
		else:
		
			print ("Actual result: ",aClass,"\n")
	
	# Increment expected_test_outputs
	j+=1
		
# Calculate accuracy of our binary perceptron

accuracy = (correctPredictions / len(testData)) * 100
print ("Classifier accuracy: ",accuracy,"%")
