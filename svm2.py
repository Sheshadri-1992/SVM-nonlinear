import matplotlib.pyplot as plt
import random 
import numpy as np
from numpy import array, dot, random
import csv
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from tabulate import tabulate
 

class_1 =[]
class_2 = []
X=[]
Y=[]

def column(matrix, i):
    return [row[i] for row in matrix]


def readInput():
	file_name = "Data_SVM.csv"

	with open(file_name) as f:
		reader = csv.reader(f)
		data = list(list(record) for record in  csv.reader(f,delimiter=','))
		f.close()
	data = data[1:]

	for record in data:
		record = [float(i) for i in record ]
		X.append(record[0:2])
		Y.append(record[2])
		if(record[2]==1):
			class_1.append(record)
		else:
			class_2.append(record)


def getScore(clf,X,Y):

	scores = []
	for i in range(0,30):
		score = cross_val_score(clf,X,Y, cv=10)
		scores.append(score)

	return np.array(scores)
    # print scores.mean()," ",scores.std()





readInput()
X= np.array(X)
Y = np.array(Y)	


max_C=0
max_d=0
max_score=0
min_std=100


w, h = 6, 11;
score_Matrix = [[0 for x in range(w)] for y in range(h)] 

sd_Matrix = [[0 for x in range(w)] for y in range(h)] 

for i in range(11):
	score_Matrix[i][0]="C="+str(i)
	sd_Matrix[i][0]="C="+str(i)


for j in range(6):
	score_Matrix[0][j]="d="+str(j)
	sd_Matrix[0][j]="d="+str(j)


for C in range(1,11):
	for d in range(1,6):
		poly_svc = SVC(kernel='poly', degree=d, C=C).fit(X, Y) # best C=20,d=4
		scores = getScore(poly_svc,X,Y)
		print "C = %d , d = %d , score = %f , sd = %f"%(C,d,scores.mean(),np.std(scores))
		score_Matrix[C][d]=scores.mean()
		sd_Matrix[C][d]=np.std(scores)

		if(scores.mean()>max_score):
			max_score=scores.mean()
			max_C=C
			max_d=d
			min_std=np.std(scores)



# print "Max",max_C
# print "Max",max_d

# 

print tabulate(score_Matrix, tablefmt="grid")

print "---------------"

print tabulate(sd_Matrix, tablefmt="grid")



poly_svc = SVC(kernel='poly', degree=max_d, C=max_C).fit(X, Y) 
clf=poly_svc


h=.02

x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

plt.plot(column(class_1,0),column(class_1,1),'ro')
plt.plot(column(class_2,0),column(class_2,1),'bo')

plt.title("Polynomial Kernel, Degree = %d, C = %d , Accuracy = %f, SD = %f "%(max_d,max_C,max_score,min_std))
plt.show()


# plt.subplot(2,2,2)

# clf=rbf_svc

# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# # Put the result into a color plot
# Z = Z.reshape(xx.shape)

# plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

# plt.plot(column(class_1,0),column(class_1,1),'ro')
# plt.plot(column(class_2,0),column(class_2,1),'bo')

# # plt.title("Poly degree = 4, C=20")
# plt.title("RBF gamma = 0.9, C=20")

# plt.show()



