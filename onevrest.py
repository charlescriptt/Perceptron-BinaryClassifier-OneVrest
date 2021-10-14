
aClass = input("choose your class, (class-1, class-2, class-3): ")
if aClass == "class-1":
    bClass = "class-2"
    cClass = "class-3"
elif aClass == "class-2":
    bClass = "class-1"
    cClass = "class-3"
elif aClass == "class-3":
    bClass = "class-1"
    cClass = "class-2"

print("class A: ", aClass)
print("class B: ", bClass)
print("class C: ", cClass)

# TRAINING PORTION

import numpy as np
from Perceptron import Perceptron
perceptron = Perceptron(4)
perceptron2 = Perceptron(4)



#------PERCEPTRON CREATION --------

#------First Binary Perceptron -----
f = open('train.data')
tI1 = []                        #training inputs for p1
np.array(tI1,dtype=float)
eO1 = []                        #expected outputs for p1

#first binary percpetron classification

for line in f:

	lines = line.rstrip()	
	linesplit = lines.split(',')
	
	if linesplit[4] == aClass:
	
		tI1.append(np.array([linesplit[0],linesplit[1],linesplit[2],linesplit[3]]))		
		eO1.append(1)
		
	elif linesplit[4] == bClass:
	
		tI1.append(np.array([linesplit[0],linesplit[1],linesplit[2],linesplit[3]]))		
		eO1.append(0)

f.close()
	
perceptron.train(tI1, eO1)



#------Secind Binary Perceptron ---- 

f = open('train.data')

tI2 = []                        #training inputs for p2
np.array(tI2,dtype=float)
eO2 = []                        #expected outputs for p2

for line in f:

	lines = line.rstrip()	
	linesplit = lines.split(',')
	
	if linesplit[4] == aClass:
	
		tI2.append(np.array([linesplit[0],linesplit[1],linesplit[2],linesplit[3]]))		
		eO2.append(1)
		
	elif linesplit[4] == cClass:
	
		tI2.append(np.array([linesplit[0],linesplit[1],linesplit[2],linesplit[3]]))		
		eO2.append(0)

f.close()
	

perceptron2.train(tI2, eO2)





# ------ read test files ----------

# ----- CASE 1 ---------

f2 = open ('test.data')

testIn1 = []   #testing input
eTestOut1 = []      #expected test output

np.array(testIn1, dtype = float)

for line in f2:                        #reads in the test array and values

	lines = line.rstrip()

	linesplit = lines.split(',')
	
	if linesplit[4] == aClass:
	
		testIn1.append(np.array([linesplit[0],linesplit[1],linesplit[2],linesplit[3]]))		
		eTestOut1.append(linesplit[4])
		
	elif linesplit[4] == bClass:
	
		testIn1.append(np.array([linesplit[0],linesplit[1],linesplit[2],linesplit[3]]))		
		eTestOut1.append(linesplit[4])
		
f2.close()

#------ CASE 2 -------

f2 = open ('test.data')

testIn2 = []   #testing input
eTestOut2 = []      #expected test output

np.array(testIn2, dtype = float)

for line in f2:                        #reads in the test array and values

	lines = line.rstrip()

	linesplit = lines.split(',')
	
	if linesplit[4] == aClass:
	
		testIn2.append(np.array([linesplit[0],linesplit[1],linesplit[2],linesplit[3]]))		
		eTestOut2.append(linesplit[4])
		
	elif linesplit[4] == cClass:
	
		testIn2.append(np.array([linesplit[0],linesplit[1],linesplit[2],linesplit[3]]))		
		eTestOut2.append(linesplit[4])
		
f2.close()










#-------- PREDICTION CHECK -------


#---------Classifier 1---------

print("PERCEPTRON CLASSIFIER 1 (A+B) : ")
print()



j = 0   #counter to hold position of operation in arrays    
truePositives = 0     #tracks correct predictions in order for later evaluation

for i in testIn1:

	print ("Data input: ",i)                          #prints data and what we should get
	print ("perceptron return", perceptron.predict(i))
	print ("Expected result: ", eTestOut1[j])   
	
	if (perceptron.predict(i) == 1):
	
		if (eTestOut1[j] == aClass):		
			truePositives += 1		
			print ("Actual result: ",aClass,"\n")
			
		else:		
			print ("Actual result: ",bClass,"\n")
		
	else:
	
		if (eTestOut1[j] == bClass):		
			truePositives += 1	
			print ("Actual result: ",bClass,"\n")
			
		else:		
			print ("Actual result: ",aClass,"\n")
	
	j+=1



accuracy = (truePositives / len(testIn1)) * 100       #accuracy calculation
print (" Percpetron 1 Accuracy= ",accuracy,"%")
print()


#------- classifier two ------

print("PERCEPTRON CLASSIFIER 2 (A+C) : ")
print()



j = 0   #counter to hold position of operation in arrays    
truePositives = 0     #tracks correct predictions in order for later evaluation


for i in testIn2:

	print ("Data input: ",i)                          #prints data and what we should get
	print ("perceptron return", perceptron2.predict(i))	
	print ("Expected result: ", eTestOut2[j])
	
	if (perceptron2.predict(i) == 1):
	
		if (eTestOut2[j] == aClass):		
			truePositives += 1		
			print ("Actual result: ",aClass,"\n")
			
		else:		
			print ("Actual result: ",cClass,"\n")
		
	else:
	
		if (eTestOut2[j] == cClass):		
			truePositives += 1	
			print ("Actual result: ",cClass,"\n")
			
		else:		
			print ("Actual result: ",aClass,"\n")
	
	j+=1



accuracy = (truePositives / len(testIn1)) * 100       #accuracy calculation
print (" Percpetron 2 Accuracy= ",accuracy,"%")

#---------- one v rest conclusion ----------



f2 = open ('test.data')

testInF = []   #testing input
eTestOutF = []      #expected test output

np.array(testInF, dtype = float)

for line in f2:                        #reads in the test array and values

	lines = line.rstrip()

	linesplit = lines.split(',')
	
	testInF.append(np.array([linesplit[0],linesplit[1],linesplit[2],linesplit[3]]))		
	eTestOutF.append(linesplit[4])
		
f2.close()



print()
print("OVR CLASSIFICATION")
print()

positives = 0
negatives = 0

classCount1 = 0
classCount2 = 0

j = 0
for i in testInF:

        print("data input: ", i)
        print ()
        print ("Expected result: ", eTestOutF[j])
        print ()
        print ("Percpetron 1: ", perceptron.predict(i))
        print ("Perceptron 2: ", perceptron2.predict(i))
       
        if(eTestOutF[j] == aClass):
              classCount1 +=1
        else:
              classCount2 +=1

        if(perceptron.predict(i) == 1):
           
              if(perceptron2.predict(i) == 1):
                  
                      print("actual result: ", aClass)
                      positives += 1

              elif(perceptron2.predict(i) == 0):
                      print("actual result: ", cClass)
                      negatives += 1

        elif(perceptron.predict(i) == 0):

              if(perceptron2.predict(i) == 1):

                      print("actual result ", bClass)
                      negatives += 1

              elif(perceptron2.predict(i) == 0):

                      print("actual result unknown")
                      negatives += 1
 
        j+=1
        print ()



accuracy = (positives / classCount1) * 100 
print("postives accuracy: ",accuracy,"%")

accuracy = (negatives / classCount2) * 100 
print("negatives accuracy: ",accuracy,"%")






