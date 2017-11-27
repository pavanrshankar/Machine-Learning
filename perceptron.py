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

####################################################################################################################

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

def getValuesforROCDET():
	train_data1, train_data2, train_data3, val_data, val_labels, valsize = getTotalData()
	a, feature_labels, s1, s2, s3 = classify(train_data1, train_data2, train_data3, val_data, val_labels, 0.5, valsize)

	print(val_labels)
	print(s1)
	print(s2)
	print(s3)
	print(feature_labels)

	return val_labels, s1, s2, s3, feature_labels

def runPerceptron(X, Y):
	e = 0.01
	w = np.zeros(X.shape[1])

	train_errors = []
	for epochs in range(1,500):
		error = 0
		for epoch in range(epochs):
			#Batch size is size of examples
			m = np.matmul(X, w)*Y

			#ind contains all ind for which dot(X, w)*Y <= 0
			ind = m<=0
			error += sum(m[m<=0])			

			#Add e*X*Y for all vectors that satisfies it
			y_ind = np.array(Y[ind]).reshape(len(Y[ind]),1)
			w += e*sum(X[ind]*y_ind)
		train_errors += [-error]
		print(epochs)
		print(-error)

	plt.plot(train_errors)
	plt.xlabel('Epoch')
	plt.ylabel('Errors')
	plt.show()

	return w	

def checkAccuracy(onevstwo, onevsthree, twovsthree, val_data, val_labels, valsize):
	res1_2 = np.matmul(val_data, onevstwo)
	res1_3 = np.matmul(val_data, onevsthree)
	res2_3 = np.matmul(val_data, twovsthree)	

	#Class of feature vectors
	class1 = [0 if ele>0 else 1 for ele in res1_2]
	score1 = np.array([ele if ele>0 else 0 for ele in res1_2])
	score2 = np.array([abs(ele) if ele<0 else 0 for ele in res1_2])

	class2 = [0 if ele>0 else 2 for ele in res1_3]
	score1 += np.array([ele if ele>0 else 0 for ele in res1_3])
	score3 = np.array([abs(ele) if ele<0 else 0 for ele in res1_3])

	class3 = [1 if ele>0 else 2 for ele in res2_3]
	score2 += np.array([ele if ele>0 else 0 for ele in res2_3])
	score3 += np.array([abs(ele) if ele<0 else 0 for ele in res2_3])

	feature_labels = []
	for i in range(len(class1)):
		cnt = Counter([class1[i], class2[i], class3[i]])
		feature_labels += [cnt.most_common(1)[0][0]]

	start = 0
	end = 0
	predicted_labels = []
	s1 = []
	s2 = []
	s3 = []
	for l in valsize:
		end += l
		cnt = Counter(feature_labels[start:end])
		predicted_labels += [cnt.most_common(1)[0][0]]
		s1 += [sum(score1[start:end])]
		s2 += [sum(score2[start:end])]
		s3 += [sum(score3[start:end])]
		start = end
	
	acc = 0
	for i in range(len(predicted_labels)):
		if(predicted_labels[i] == val_labels[i]):
			acc = acc + 1

	print(acc/len(predicted_labels))
	return s1, s2, s3, predicted_labels

def getScoresPerceptron():
	train_data1, train_data2, train_data3, val_data, val_labels, valsize = getTotalData()

	#Adding column of 1's to val
	val_data = np.append(np.ones([len(val_data),1]),val_data,1)

	#Weights
	onevstwo = np.array([3154.08, 18.6855023, 58.1954264, 116.341563, 48.5546255, 83.0637857, 64.2289094, -72.4866562, 0.569607923, 23.8893155, 357.889769, 1385.94597, 1280.02211, -2453.5813, -2340.69732, 955.879011, 73.4534793, 3703.14673, 148.624875, -476.987958, 1.02558147, 15.7630537, 287.743611, 169.809102])
	onevsthree = np.array([12032.35, -15.9770705, 93.45819, 65.9520967, -13.0941199, -27.2581075, 9.14746635, -7.35309355, 34.2859471, 89.2110206, 225.292939, 4345.64899, 2909.0324, -7205.77104, -6692.59046, 2070.87771, 1730.2928, 10401.7587, 135.397439, -84.7509231, -97.8085749, 38.3034375, 114.81486, -97.1727935])
	twovsthree = np.array([-6699.77, -5.11514171, -155.354457, -130.583909, -52.0183737, -71.1286253, -5.35620712, 15.5406812, -102.160611, -8.81614931, -452.455479, -2211.21714, -2449.21747, 5898.50581, 5940.73192, -3627.5362, -2039.41408, -9509.90047, 355.267051, 265.397856, -470.154202, -247.004842, -132.172301, -115.987605])
	
	score1, score2, score3, predicted_labels = checkAccuracy(onevstwo, onevsthree, twovsthree, val_data, val_labels, valsize)

	return val_labels, np.array(score1), np.array(score2), np.array(score3), predicted_labels

if __name__ == '__main__':
	#train data - rows; val_data - rows; val_lables - per_image; valsize - [no of features in file1, no if features in file2,....]
	train_data1, train_data2, train_data3, val_data, val_labels, valsize = getTotalData()

	#Adding column of 1's to train
	train_data1 = np.append(np.ones([len(train_data1),1]),train_data1,1)
	train_data2 = np.append(np.ones([len(train_data2),1]),train_data2,1)
	train_data3 = np.append(np.ones([len(train_data3),1]),train_data3,1)

	y_labels = np.array(len(train_data1)*[-1] + len(train_data2)*[-1] + len(train_data3)*[1])
	train_data = np.concatenate([train_data1, train_data2, train_data3], axis=0)
	onevstwo = runPerceptron(train_data, y_labels)
	print(onevstwo)
	
	'''
	#1 vs 1 Perceptrons
	########################################################################################
	#onevstwo
	y_labels = np.array(len(train_data1)*[1] + len(train_data2)*[-1])
	train_data = np.concatenate([train_data1, train_data2], axis=0)
	#ind = np.arange(len(y_labels))
	#np.random.shuffle(ind)
	#x = train_data[ind]
	#y = y_labels[ind]
	onevstwo = runPerceptron(train_data, y_labels)
	print(onevstwo)
	
	#########################################################################################
	#oneVsthird
	y_labels = np.array(len(train_data1)*[1] + len(train_data3)*[-1])
	train_data = np.concatenate([train_data1, train_data3], axis=0)
	onevsthird = runPerceptron(train_data, y_labels)
	print(onevsthird)
	
	#########################################################################################
	#secVsthird
	y_labels = np.array(len(train_data2)*[1] + len(train_data3)*[-1])
	train_data = np.concatenate([train_data2, train_data3], axis=0)
	secVsthird = runPerceptron(train_data, y_labels)
	print(secVsthird)

	#########################################################################################
	'''
	
	#Adding column of 1's to val
	val_data = np.append(np.ones([len(val_data),1]),val_data,1)

	#Weights
	onevstwo = np.array([3154.08, 18.6855023, 58.1954264, 116.341563, 48.5546255, 83.0637857, 64.2289094, -72.4866562, 0.569607923, 23.8893155, 357.889769, 1385.94597, 1280.02211, -2453.5813, -2340.69732, 955.879011, 73.4534793, 3703.14673, 148.624875, -476.987958, 1.02558147, 15.7630537, 287.743611, 169.809102])
	onevsthree = np.array([12032.35, -15.9770705, 93.45819, 65.9520967, -13.0941199, -27.2581075, 9.14746635, -7.35309355, 34.2859471, 89.2110206, 225.292939, 4345.64899, 2909.0324, -7205.77104, -6692.59046, 2070.87771, 1730.2928, 10401.7587, 135.397439, -84.7509231, -97.8085749, 38.3034375, 114.81486, -97.1727935])
	twovsthree = np.array([-6699.77, -5.11514171, -155.354457, -130.583909, -52.0183737, -71.1286253, -5.35620712, 15.5406812, -102.160611, -8.81614931, -452.455479, -2211.21714, -2449.21747, 5898.50581, 5940.73192, -3627.5362, -2039.41408, -9509.90047, 355.267051, 265.397856, -470.154202, -247.004842, -132.172301, -115.987605])
	
	score1, score2, score3, predicted_labels = checkAccuracy(onevstwo, onevsthree, twovsthree, val_data, val_labels, valsize)