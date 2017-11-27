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
from scipy import misc, linalg
from math import log
from sklearn.preprocessing import MinMaxScaler
from statistics import mode
from collections import Counter
from scipy.misc import logsumexp
from scipy import cluster

##############################
#Within class covariance normalization
def wccn(A, m):
	M = linalg.sqrtm(np.linalg.inv(np.cov(A.T)))
	T = np.matmul(M, (A-m).T).T
	return T, M

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
	
def getTotalData(norm=True):
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

		if(norm):
			totaldata += [normalize(data)]
		else:
			totaldata += [data]

		filelen += [l]

	train_data1, val_data1, test_data1, valsize1 = splitData(totaldata[0], filelen[0])
	train_data2, val_data2, test_data2, valsize2 = splitData(totaldata[1], filelen[1])
	train_data3, val_data3, test_data3, valsize3 = splitData(totaldata[2], filelen[2])

	val_data = [point for point in val_data1] + [point for point in val_data2] + [point for point in val_data3]
	val_labels = len(valsize1)*[0] + len(valsize2)*[1] + len(valsize3)*[2]
	valsize = valsize1 + valsize2 + valsize3

	return np.array(train_data1), np.array(train_data2), np.array(train_data3), np.array(val_data), np.array(val_labels), valsize

def lda(train_data1, train_data2, train_data3, norm=True):
	if(norm):
		#Initial means before wccn
		mean = np.array([np.mean(np.array(train_data1),axis=0), np.mean(np.array(train_data2),axis=0), np.mean(np.array(train_data3),axis=0)])

		#WCCN
		train_data1, covinv1 = wccn(train_data1, mean[0])
		train_data2, covinv2 = wccn(train_data2, mean[1])
		train_data3, covinv3 = wccn(train_data3, mean[2])

		train_data1 = train_data1 + np.matmul(covinv1, mean[0].T).T
		train_data2 = train_data2 + np.matmul(covinv2, mean[1].T).T
		train_data3 = train_data3 + np.matmul(covinv3, mean[2].T).T

		covall = (covinv1 + covinv2 + covinv3) / 3
		covall = np.diag(np.diag(covall))

	#Recalculate mean
	mean = np.array([np.mean(np.array(train_data1),axis=0), np.mean(np.array(train_data2),axis=0), np.mean(np.array(train_data3),axis=0)])

	#Sw - Recalculate within class covariance matrix
	Sw = np.matmul((train_data1-mean[0]).T, train_data1-mean[0]) + np.matmul((train_data2-mean[1]).T, train_data2-mean[1]) + np.matmul((train_data3-mean[2]).T, train_data3-mean[2])

	#Overall mean	
	mean_over = np.mean(mean,axis=0)
	mean_over = mean_over.reshape(1,mean.shape[1])
	
	#Sb - between class covariance matrix
	Sb = len(train_data1)*np.matmul((mean[0] - mean_over).T, (mean[0] - mean_over)) + len(train_data2)*np.matmul((mean[1] - mean_over).T, (mean[1] - mean_over)) + len(train_data3)*np.matmul((mean[2] - mean_over).T, (mean[2] - mean_over))

	#Eigen decomposition
	Swinv = np.linalg.inv(Sw)
	eigvals, eigvecs = np.linalg.eig(np.matmul(Swinv,Sb))
	sorind = np.argsort(np.array(-eigvals.real))

	#Choosing top 2(C-1) eigen vectors for projection
	eigvecs = eigvecs[sorind[0:2]].real
	
	#Projecting training data
	projtrain_data1 = np.matmul(train_data1, eigvecs.T)
	projtrain_data2 = np.matmul(train_data2, eigvecs.T)
	projtrain_data3 = np.matmul(train_data3, eigvecs.T)

	axes = plt.gca()
	axes.set_axisbelow(True)
	axes.minorticks_on()
	axes.grid(which='major', linestyle='-', linewidth='0.5', color='black')	

	plt.scatter([ele[0] for ele in projtrain_data1], [ele[1] for ele in projtrain_data1], c='r',s=0.5,label='Coast')
	plt.scatter([ele[0] for ele in projtrain_data2], [ele[1] for ele in projtrain_data2], c='b',s=0.5,label='Mountain')
	plt.scatter([ele[0] for ele in projtrain_data3], [ele[1] for ele in projtrain_data3], c='g',s=0.5,label='Tall Building')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()

	cov1  = np.cov(projtrain_data1.T)
	cov2  = np.cov(projtrain_data2.T)
	cov3  = np.cov(projtrain_data3.T)

	mu1 = np.mean(projtrain_data1,axis=0)
	mu2 = np.mean(projtrain_data2,axis=0)
	mu3 = np.mean(projtrain_data3,axis=0)

	if(norm):
		return eigvecs, cov1, cov2, cov3, mu1, mu2, mu3, covall
	else:
		return eigvecs, cov1, cov2, cov3, mu1, mu2, mu3

def withoutWCCNAccuracy(val_data, eigvecs, cov1, cov2, cov3, mu1, mu2, mu3, valsize, val_labels):
	proj_data = np.matmul(val_data, eigvecs.T)

	score1 = multivariate_normal.pdf(proj_data, mean=mu1, cov=cov1)
	score2 = multivariate_normal.pdf(proj_data, mean=mu2, cov=cov2)
	score3 = multivariate_normal.pdf(proj_data, mean=mu3, cov=cov3)

	#Accuracy
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
	predicted_labels = np.argmax([s1,s2,s3],axis=0)

	acc = 0
	for i in range(len(predicted_labels)):
		if(predicted_labels[i] == val_labels[i]):
			acc = acc + 1

	print(acc/len(predicted_labels))

	return np.array(s1), np.array(s2), np.array(s3), predicted_labels

def getScoresLDA():
	train_data1, train_data2, train_data3, val_data, val_labels, valsize = getTotalData(True)
	eigvecs, cov1, cov2, cov3, mu1, mu2, mu3 = lda(train_data1, train_data2, train_data3, False)
	score1, score2, score3, predicted_labels = withoutWCCNAccuracy(val_data, eigvecs, cov1, cov2, cov3, mu1, mu2, mu3, valsize, val_labels)

	return val_labels, score1, score2, score3, predicted_labels

if __name__ == '__main__':
	#train_data1, train_data2, train_data3, val_data, val_labels, valsize = getTotalData(False)
	#eigvecs, cov1, cov2, cov3, mu1, mu2, mu3, covall = lda(train_data1, train_data2, train_data3, True)
	#val_data = np.matmul(covall, val_data.T).T
	#withoutWCCNAccuracy(val_data, eigvecs, cov1, cov2, cov3, mu1, mu2, mu3, valsize, val_labels)

	train_data1, train_data2, train_data3, val_data, val_labels, valsize = getTotalData(True)
	eigvecs, cov1, cov2, cov3, mu1, mu2, mu3 = lda(train_data1, train_data2, train_data3, False)
	s1, s2, s3, predicted_labels = withoutWCCNAccuracy(val_data, eigvecs, cov1, cov2, cov3, mu1, mu2, mu3, valsize, val_labels)
	