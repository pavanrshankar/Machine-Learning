import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from os import listdir
from os.path import isfile, join
from scipy.stats import mode
from sklearn import metrics
from scipy.stats import norm
from scipy import misc
from math import log
from sklearn.preprocessing import MinMaxScaler
from statistics import mode
from collections import Counter
from scipy.misc import logsumexp

################################################################################

def normalize(A):
	A = A / np.std(A, axis=0)
	return A	

def getFileData(path):
	df = pd.read_csv(path, header=None, delim_whitespace=True)
	data = np.array(df)
	return data

#Splitting images into training, val and testing
def splitData(data, filelen, train_ratio=0.7, val_ratio=0.15):
	no_files = len(filelen)
	train_size = (int)(no_files * train_ratio)
	val_size = (int)(no_files * val_ratio)
	test_size = no_files - train_size - val_size

	total_train_feat = sum(filelen[0:train_size])
	total_val_feat = sum(filelen[train_size:train_size+val_size])

	train_data = data[0:total_train_feat,:]
	val_data = data[total_train_feat:total_val_feat+total_train_feat,:]
	test_data = data[total_val_feat+total_train_feat:,:]

	return train_data, val_data, test_data, filelen[train_size:train_size+val_size]

def getNormalizedScores(score1,score2,score3):
	total = score1 + score2 + score3
	score1 = score1/total
	score2 = score2/total
	score3 = score3/total

	return score1, score2, score3

def classify(train_data1, train_data2, train_data3, val_data, val_labels, h, valsize):
	score1 = []
	score2 = []
	score3 = []

	#Calculate gaussian distance between val point and each of train_data
	for p in val_data:
		score1 += [logsumexp(np.sum((p-train_data1)*(p-train_data1) ,axis=1) / (-2*h*h))]
		score2 += [logsumexp(np.sum((p-train_data2)*(p-train_data2) ,axis=1) / (-2*h*h))]
		score3 += [logsumexp(np.sum((p-train_data3)*(p-train_data3) ,axis=1) / (-2*h*h))]
	
	score1 = np.array(score1)
	score2 = np.array(score2)
	score3 = np.array(score3)

	print(score1)
	print(score2)
	print(score3)

	#Combining feature vectors using likelihood
	start = 0
	end = 0
	s1 = []
	s2 = []
	s3 = []
	for l in valsize:
		end += l
		s1 += [sum(score1[start:end])]
		s2 += [sum(score2[start:end])]
		s3 += [sum(score3[start:end])]
		start = end
	feature_labels = np.argmax([s1,s2,s3],axis=0)

	acc = 0
	for i in range(len(feature_labels)):
		if(feature_labels[i] == val_labels[i]):
			acc = acc + 1
	
	return acc/len(feature_labels), feature_labels, np.array(s1), np.array(s2), np.array(s3)
	
def getTotalData():
	path = '/Users/pavan.ravishankar/Downloads/pr5/data/'
	types = ['coast/', 'mountain/', 'tallbuilding/']

	#[coast, mountain, tallbuilding]
	totaldata = []
	filelen = []

	for tp in types:
		opath = path + tp
		files = [join(opath, f) for f in listdir(opath) if isfile(join(opath, f))]
		
		data = getFileData(files[0])
		l = [len(data)]
		for fname in files:
			temp = getFileData(fname)
			l += [len(temp)]
			data = np.concatenate((data, temp), axis=0)

		totaldata += [normalize(data)]
		filelen += [l]

	train_data1, val_data1, test_data1, valsize1 = splitData(totaldata[0], filelen[0])
	train_data2, val_data2, test_data2, valsize2 = splitData(totaldata[1], filelen[1])
	train_data3, val_data3, test_data3, valsize3 = splitData(totaldata[2], filelen[2])

	val_data = [point for point in val_data1] + [point for point in val_data2] + [point for point in val_data3]
	val_labels = len(valsize1)*[0] + len(valsize2)*[1] + len(valsize3)*[2]
	
	valsize = valsize1 + valsize2 + valsize3

	return np.array(train_data1), np.array(train_data2), np.array(train_data3), np.array(val_data), np.array(val_labels), valsize

def getScoresParzen():
	train_data1, train_data2, train_data3, val_data, val_labels, valsize = getTotalData()
	a, feature_labels, s1, s2, s3 = classify(train_data1, train_data2, train_data3, val_data, val_labels, 0.5, valsize)

	print(val_labels)
	print(s1)
	print(s2)
	print(s3)
	print(feature_labels)

	return val_labels, s1, s2, s3, feature_labels

if __name__ == '__main__':
	train_data1, train_data2, train_data3, val_data, val_labels, valsize = getTotalData()

	#Choosing h values
	acc = []
	h_values = []
	for h in np.arange(0.05,30,5):
		a, feature_labels, s1, s2, s3 = classify(train_data1, train_data2, train_data3, val_data, val_labels, h, valsize)
		h_values += [h]
		acc += [a]
		print(a)

	print(acc)	

		
	axes = plt.gca()
	axes.set_axisbelow(True)
	axes.minorticks_on()
	axes.grid(which='major', linestyle='-', linewidth='0.5', color='red')	
	axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
	
	plt.plot(np.arange(0.05,30,5), acc, c='r')
	plt.xlabel('h value')
	plt.ylabel('Validation accuracy')
	plt.show()

