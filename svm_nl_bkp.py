import numpy as np
import csv
import matplotlib.pyplot as plt
from numpy.linalg import inv
import random
import sys
import plot
import k_cross_validation
from sklearn import svm
from tabulate import tabulate

#read data and populate into files
def loadSamples(samples,fileName,classType):
	
	reader = np.genfromtxt(fileName,delimiter=',')
	x = list(reader)
	samples = []

	for i in range(len(x)):
		
		y = x[i]
		if(y[-1]==classType):
			samples.append(y[0:2])


	return samples

dataset = "Data_SVM.csv"

class_samples_1 = []
class_samples_2 = []
class_samples_1 = loadSamples(class_samples_1,dataset,-1)
class_samples_2 = loadSamples(class_samples_2,dataset,1)

print "The number of entries in class 1 are ",len(class_samples_1)
print "The number of entries in class 2 are ",len(class_samples_2)

# prepare array for SVM classification 
# required data type is 
# X = [[0, 0], [1, 1]] -> y = [0, 1] # the points are {p1,p2} -> {class 0, class 1}

samples_data_svm_format = []

#items in class samples 1
for x in class_samples_1:
	temp_item = []

	for y in x:
		temp_item.append(y)		

	temp_item.append(-1)
	samples_data_svm_format.append(temp_item)
	

#items in class samples 2
for x in class_samples_2:
	temp_item = []

	for y in x:
		temp_item.append(y)		
	
	temp_item.append(1)
	samples_data_svm_format.append(temp_item)
	
# samples_data_svm_format = np.array(samples_data_svm_format)

print "sample data is ",samples_data_svm_format[0]

#actual work
num_epoch = 30
accList = []
x1 = [x[0] for x in class_samples_1]
y1 = [y[1] for y in class_samples_1]

x2 = [x[0] for x in class_samples_2]
y2 = [y[1] for y in class_samples_2]


max_C=0
max_d=0
max_score=0
min_std=100

w, h = 11, 12;
score_Matrix = [[0 for x in range(w)] for y in range(h)] 

sd_Matrix = [[0 for x in range(w)] for y in range(h)] 

for i in range(1,12):
	score_Matrix[i][0]="C="+str(i+9)
	sd_Matrix[i][0]="C="+str(i+9)

g=0.5
for j in range(1,11):
	score_Matrix[0][j]="rho="+str(g)
	sd_Matrix[0][j]="rho="+str(g)
	g=g+0.1


D = 1
gammaList = [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4]
kernelType = 'rbf'

print kernelType

for C in range(10,21):
	for j in range(len(gammaList)):

		accList = []

		for epochs in range(num_epoch):

			random.shuffle(samples_data_svm_format) # data is shuffled now, we can perform 10 cross validation now

			label1 = 0
			label2 = 0
			for j in range(len(samples_data_svm_format)):
				if(samples_data_svm_format[j][-1]==-1):
					label1 = label1 + 1
				else:
					label2 = label2 + 1

			ecpochAccuracy = 0
			for i in range(10):
				# print "here"
				ecpochAccuracy = ecpochAccuracy + k_cross_validation.prepare_train_test_data(samples_data_svm_format,i,10,x1,y1,x2,y2,C,D,gammaList[i],kernelType)
			# print "epoch ",epochs, " Accuracy ",ecpochAccuracy/float(10)
			accList.append(ecpochAccuracy/float(10))

		accList = np.array(accList)
		score_Matrix[C][j]=accList.mean()
		sd_Matrix[C][j]=np.std(accList)

		print "here"

		if(accList.mean()>max_score):
			max_score=accList.mean()
			max_C=C
			max_d=gammaList[i]
			min_std=np.std(accList)


print tabulate(score_Matrix, tablefmt="grid")

print "---------------"

print tabulate(sd_Matrix, tablefmt="grid")			

		

# plotting of graph
# Class 1 samples
# x1 = [x[0] for x in class_samples_1]
# y1 = [y[1] for y in class_samples_1]
# plot.loadGraph(x1,y1,'ro')


# # Class 2 samples
# x1 = [x[0] for x in class_samples_2]
# y1 = [y[1] for y in class_samples_2]

# plot.loadGraph(x1,y1,'bo')

# plot.showGraph()
