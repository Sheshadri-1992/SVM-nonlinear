#!/usr/bin/python

import matplotlib.pyplot as plt

def loadGraph(x,y,color):
	plt.plot(x,y,color) 
	plt.xlabel("X")
	plt.ylabel("Y")

def showGraph():
	plt.show()