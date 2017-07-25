#!/usr/bin/python
from sklearn import svm
import sys
import numpy as np
from sklearn.svm import SVC
import time
import matplotlib.pyplot as py

def trainSVM(train_data,test_data,x1,y1,x2,y2,C_arg,d_arg,gamma_arg,kernel_type):

	# prepare train data in the svm acceptable format
	train_x = []
	train_labels = []

	for x in train_data:
		item = []
		for i in range(0,len(x)-1):
			item.append(x[i])

		train_x.append(item)
		train_labels.append(x[-1])

	
	classifier = svm.SVC(C=12, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma=0.8, kernel=kernel_type,
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
	
	# time.sleep(60)
	C_val = C_arg #9.91365692002
	D_val = d_arg
	gamma_val = gamma_arg
	
	
	if kernel_type=='poly':
		print "POLY"
	elif kernel_type=='rbf':
		print "RBF"

	classifier.fit(train_x,train_labels)
	# SVC(C=12, cache_size=200, class_weight=None, coef0=0.0,
 #    decision_function_shape=None, degree=3, gamma=0.8, kernel=kernel_type,
 #    max_iter=-1, probability=False, random_state=None, shrinking=True,
 #    tol=0.001, verbose=False)


	# prepare test data in the svm acceptable format
	test_x = []	
	test_labels = []
	# print len(test_data)
	for x in test_data:
		item = []
		for i in range(0,len(x)-1):
			item.append(x[i])
		
		test_x.append(item)
		test_labels.append(x[-1])

	i = 0
	success = 0
	test_x = np.array(test_x)

	
	X = [x[0] for x in test_x] + [x[0] for x in train_x]
	Y = [x[1] for x in test_x] + [x[1] for x in train_x] 
	X = np.array(X)
	Y = np.array(Y)
	for point in test_x:
		# point = np.array(point).reshape(1,-1)
		point = [list(point)]

		prediction = classifier.predict(point)
		# print " prediction is ",prediction," test labels are ",test_labels[i]
		
		if(prediction[-1]==test_labels[i]):
			success = success+1
		i = i + 1
		
	accuracy = success/float(len(test_x))

	# print X.shape,Y.shape
	h = 0.02
	x_min = -2
	x_max =  2
	y_min = -2
	y_max =  2	
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
	Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

	#this is to print graph
	Z = Z.reshape(xx.shape)
	print "(correct : ", success,")/ (Total ",len(test_x), ") The accuracy is ",accuracy
	# py.contourf(Z, cmap=py.cm.coolwarm, alpha=0.8)
	py.title("SVM classifier rbf kernel value :C = 12, rho = 0.8") #type for rbf or poly
	py.plot(x1,y1,'go') 
	py.plot(x2,y2,'bo') 
	py.contour(xx, yy, Z, cmap=py.cm.Paired)
	py.scatter(X, Y, c='y', cmap=py.cm.coolwarm)
	py.xlabel('X axis')
	py.ylabel('Y axis')
	py.xlim(x_min, y_max)
	py.ylim(y_min, y_max)
	py.xticks(())
	py.yticks(())	
	py.show() #uncomment to show the graph

	
	
	return accuracy
	

def prepare_train_test_data(k_cross_samples,skip_number,k,x1,y1,x2,y2,C_arg,d_arg,gamma_arg,kernel_type):
	
	train_data = []
	test_data = []
	
	
	for i in range(len(k_cross_samples)):
		if( i%k == skip_number):
			test_data.append(k_cross_samples[i])
		else:
			train_data.append(k_cross_samples[i])

	accuracy = trainSVM(train_data,test_data,x1,y1,x2,y2,C_arg,d_arg,gamma_arg,kernel_type)

	del train_data[:]
	del test_data[:]

	return accuracy


	# # svr_rbf = SVR(kernel='poly', C=1e3, degree=2) #kernel='poly', C=1e3, degree=2
	# svr_rbf = SVR(kernel='rbf', C=1, gamma=0.9)

	#kernel='rbf', C=1, gamma=0.5
