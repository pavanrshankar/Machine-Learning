import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import sys
import argparse
import os
from scipy.stats import multivariate_normal
from scipy.interpolate import interp1d
from scipy.stats import norm

#############################################Helper functions#####################################################################		
def getData(path):
	df = pd.read_csv(path, header=None, delim_whitespace=True)
	data = np.array(df)
	return data


def getDataFromFile(fname):
	if os.path.isdir(fname):
		cls_files=[os.path.join(fname, f) for f in os.listdir(fname) if f!='.DS_Store']

		outp = np.ones((1000, 1))

		class1 = np.concatenate((getData(cls_files[0]), outp), axis=1) 
		class2 = np.concatenate((getData(cls_files[1]), 2*outp), axis=1) 
		class3 = np.concatenate((getData(cls_files[2]), 3*outp), axis=1)

	else:
		outp = np.ones((500, 1))	

		data = getData(fname)

		class1 = np.concatenate((data[0:500,:], outp), axis=1) 
		class2 = np.concatenate((data[500:1000,:], 2*outp), axis=1) 
		class3 = np.concatenate((data[1000:1500,:], 3*outp), axis=1)

	data = np.concatenate((class1, class2, class3), axis=0)
	return data

#############################################Prediction Functions#################################################################
def find_descriminant(x,mu,cov,prior):
	d=2
	maha=(np.dot(np.dot((x-mu).T,np.linalg.inv(cov)),(x-mu)))/2
	const_term=np.log(2*np.pi)*(d/2)
	sig_term=np.log(np.linalg.det(cov))/2
	prior_term=np.log(prior)
	g=-maha-const_term-sig_term+prior_term
	return g

def find_prediction(x, mu1, mu2, mu3, cov1, cov2, cov3, thr1, thr2, thr3):
	g1=find_descriminant(x,mu1,cov1,0.3)[0][0] + thr1 
	g2=find_descriminant(x,mu2,cov2,0.3)[0][0] + thr2
	g3=find_descriminant(x,mu3,cov3,0.3)[0][0] + thr3
	g_list=np.array((g1,g2,g3))
	return g_list.argmax()+1

def confusion_matrix(test, mu1, mu2, mu3, cov1, cov2, cov3, thr1, thr2, thr3):
	conf_matrix={}
	conf_matrix[1]=[0,0,0,0]
	conf_matrix[2]=[0,0,0,0]
	conf_matrix[3]=[0,0,0,0]

	for i in range(len(test)):
		x=test[i][:2]
		x=x.reshape((x.shape[0],1))
		act_target=test[i][-1]
		conf_matrix[act_target][0]=conf_matrix[act_target][0]+1
		pred = find_prediction(x, mu1, mu2, mu3, cov1, cov2, cov3, thr1, thr2, thr3)
		conf_matrix[act_target][pred]=conf_matrix[act_target][pred]+1

	tp1=conf_matrix[1][1]
	tp2=conf_matrix[2][2]
	tp3=conf_matrix[3][3]
	fp1=conf_matrix[2][1]+conf_matrix[3][1]
	fp2=conf_matrix[1][2]+conf_matrix[3][2]
	fp3=conf_matrix[1][3]+conf_matrix[2][3]
	tn1=(conf_matrix[2][0]+conf_matrix[3][0])-fp1
	tn2=(conf_matrix[1][0]+conf_matrix[3][0])-fp2
	tn3=(conf_matrix[1][0]+conf_matrix[2][0])-fp3

	fn1=conf_matrix[1][2]+conf_matrix[1][3]
	fn2=conf_matrix[2][1]+conf_matrix[2][3]
	fn3=conf_matrix[3][1]+conf_matrix[3][2]
	total=conf_matrix[1][0]+conf_matrix[2][0]+conf_matrix[3][0]

	acc1=(tp1+tn1)/float(total)
	acc2=(tp2+tn2)/float(total)
	acc3=(tp3+tn3)/float(total)
	avg_acc=((acc1+acc2+acc3)/3.0)*100
	
	tpr1=tp1/float(tp1+fn1)
	tpr2=tp2/float(tp2+fn2)
	tpr3=tp3/float(tp3+fn3)

	fpr1=fp1/float(fp1+tn1)
	fpr2=fp2/float(fp2+tn2)
	fpr3=fp3/float(fp3+tn3)

	mr1=(conf_matrix[1][2]+conf_matrix[1][3])/float(conf_matrix[1][0])
	mr2=(conf_matrix[2][1]+conf_matrix[2][3])/float(conf_matrix[2][0])
	mr3=(conf_matrix[3][1]+conf_matrix[3][2])/float(conf_matrix[3][0])
		
	return conf_matrix,avg_acc,tpr1,tpr2,tpr3,fpr1,fpr2,fpr3,mr1,mr2,mr3

#################################################Plot Functions####################################################################
def plotSameBayesCovariance(x, X, Y, val, rv1, rv2, rv3, mu1, mu2, mu3, cov1, cov2, cov3):
	axes = plt.gca()
	axes.set_xlim([-10,50])
	axes.set_ylim([-10,50])

	c1 = plt.contour(X,Y,rv1.pdf(val), colors='r')
	plt.clabel(c1, fmt = '%0.6f')
	c2 = plt.contour(X,Y,rv2.pdf(val), colors='b')
	plt.clabel(c2, fmt = '%0.6f')
	c3 = plt.contour(X,Y,rv3.pdf(val), colors='g')
	plt.clabel(c3, fmt = '%0.6f')

	C = (cov1 + cov2 + cov3) / 3
		
	w = np.matmul(np.linalg.inv(C), (mu1 - mu2))
	pc1 = 0.3
	pc2 = 0.3
	x0 = (mu1 + mu2) / 2 - np.log(pc1/pc2) * (mu1 - mu2) / np.matmul(np.matmul((mu1 - mu2).T, np.linalg.inv(C)), (mu1 - mu2))
	ybc = (np.matmul(w.T, x0) - w[0]*x) / w[1]
	plt.scatter(x,ybc,c='pink', label='Decision surface for Red and Blue')
		
	w = np.matmul(np.linalg.inv(C), (mu2 - mu3))
	pc2 = 0.3
	pc3 = 0.3
	x0 = (mu2 + mu3) / 2 - np.log(pc2/pc3) * (mu2 - mu3) / np.matmul(np.matmul((mu2 - mu3).T, np.linalg.inv(C)), (mu2 - mu3))
	ybc = (np.matmul(w.T, x0) - w[0]*x) / w[1]
	plt.scatter(x,ybc,c='magenta', label='Decision surface for Blue and Green')
		
	w = np.matmul(np.linalg.inv(C), (mu1 - mu3))
	pc1 = 0.3
	pc3 = 0.3
	x0 = (mu1 + mu3) / 2 - np.log(pc1/pc3) * (mu1 - mu3) / np.matmul(np.matmul((mu1 - mu3).T, np.linalg.inv(C)), (mu1 - mu3))
	ybc = (np.matmul(w.T, x0) - w[0]*x) / w[1]
	plt.scatter(x,ybc,c='green', label='Decision surface for Red and Green')
	
	plt.xlabel("x")
	plt.ylabel("y")
	plt.legend(loc='upper left')
	plt.show()

def plotDiffBayesCovariance(x, X, Y, val, rv1, rv2, rv3, mu1, mu2, mu3, cov1, cov2, cov3):
	axes = plt.gca()
	axes.set_xlim([-10,50])
	axes.set_ylim([-10,50])

	c1 = plt.contour(X,Y,rv1.pdf(val), colors='r')
	plt.clabel(c1, fmt = '%0.6f')
	c2 = plt.contour(X,Y,rv2.pdf(val), colors='b')
	plt.clabel(c2, fmt = '%0.6f')
	c3 = plt.contour(X,Y,rv3.pdf(val), colors='g')
	plt.clabel(c3, fmt = '%0.6f')

	pc1 = 0.3
	pc2 = 0.3
	pc3 = 0.3

	c1_inv = np.linalg.inv(cov1)
	c2_inv = np.linalg.inv(cov2)
	c3_inv = np.linalg.inv(cov3)

	W1 = -0.5 * c1_inv
	W2 = -0.5 * c2_inv
	W3 = -0.5 * c3_inv

	w1 = np.matmul(c1_inv, mu1)
	w2 = np.matmul(c2_inv, mu2)
	w3 = np.matmul(c3_inv, mu3)
	print(w1.shape)

	w01 = -0.5 * np.matmul(np.matmul(mu1.T, c1_inv), mu1) - 0.5 * np.log(np.linalg.det(cov1)) + np.log(pc1)
	w02 = -0.5 * np.matmul(np.matmul(mu2.T, c2_inv), mu2) - 0.5 * np.log(np.linalg.det(cov2)) + np.log(pc2)
	w03 = -0.5 * np.matmul(np.matmul(mu3.T, c3_inv), mu3) - 0.5 * np.log(np.linalg.det(cov3)) + np.log(pc3)
		
	a = W2[1][1] - W1[1][1]
	b1 = w2[1][0] - w1[1][0]
	b2 = W2[1][0] + W2[0][1] - W1[1][0] - W1[0][1]
	c1 = W2[0][0] - W1[0][0]
	c2 = w2[0][0] - w1[0][0]
	c3 = w02 - w01
	x_res = []
	y_res = []
	for p in x:
		b = b1 + p * b2
		c = p * p * c1 + p * c2 + c3 
		x_res += [p, p]
		y_res += np.roots([a,b,c]).tolist()
	plt.scatter(x_res, y_res, c='black', label='Decision surface for Red and Blue')	
		
	a = W3[1][1] - W2[1][1]
	b1 = w3[1][0] - w2[1][0]
	b2 = W3[1][0] + W3[0][1] - W2[1][0] - W2[0][1]
	c1 = W3[0][0] - W2[0][0]
	c2 = w3[0][0] - w2[0][0]
	c3 = w03 - w02
	x_res = []
	y_res = []
	for p in x:
		b = b1 + p * b2
		c = p * p * c1 + p * c2 + c3 
		x_res += [p, p]
		y_res += np.roots([a,b,c]).tolist()
	plt.scatter(x_res, y_res, c='green', label='Decision surface for Blue and Green')
		
	a = W1[1][1] - W3[1][1]
	b1 = w1[1][0] - w3[1][0]
	b2 = W1[1][0] + W1[0][1] - W3[1][0] - W3[0][1]
	c1 = W1[0][0] - W3[0][0]
	c2 = w1[0][0] - w3[0][0]
	c3 = w01 - w03
	x_res = []
	y_res = []
	for p in x:
		b = b1 + p * b2
		c = p * p * c1 + p * c2 + c3 
		x_res += [p, p]
		y_res += np.roots([a,b,c]).tolist()
	plt.scatter(x_res, y_res, c='orange', label='Decision surface for Red and Green')

	plt.xlabel("x")
	plt.ylabel("y")
	plt.legend(loc='upper left')
	plt.show()

def plotNaiveSigma2(x, X, Y, val, rv1, rv2, rv3, mu1, mu2, mu3, cov1, cov2, cov3):
	axes = plt.gca()
	axes.set_xlim([-10,50])
	axes.set_ylim([-10,50])

	c1 = plt.contour(X,Y,rv1.pdf(val), colors='r')
	plt.clabel(c1, fmt = '%0.6f')
	c2 = plt.contour(X,Y,rv2.pdf(val), colors='b')
	plt.clabel(c2, fmt = '%0.6f')
	c3 = plt.contour(X,Y,rv3.pdf(val), colors='g')
	plt.clabel(c3, fmt = '%0.6f')

	C_naive_diag=np.array((np.diag(cov1),np.diag(cov2),np.diag(cov3))).mean() * np.eye(cov1.shape[0])
	
	w = np.matmul(np.linalg.inv(C_naive_diag), (mu1 - mu2))
	pc1 = 0.3
	pc2 = 0.3
	x0 = (mu1 + mu2) / 2 - np.log(pc1/pc2) * (mu1 - mu2) / np.matmul(np.matmul((mu1 - mu2).T, np.linalg.inv(C_naive_diag)), (mu1 - mu2))
	ybc = (np.matmul(w.T, x0) - w[0]*x) / w[1]
	plt.scatter(x,ybc,c='brown', label='Naive Decision surface for Red and Blue with sigma^2*I')
		
	w = np.matmul(np.linalg.inv(C_naive_diag), (mu2 - mu3))
	pc2 = 0.3
	pc3 = 0.3
	x0 = (mu2 + mu3) / 2 - np.log(pc2/pc3) * (mu2 - mu3) / np.matmul(np.matmul((mu2 - mu3).T, np.linalg.inv(C_naive_diag)), (mu2 - mu3))
	ybc = (np.matmul(w.T, x0) - w[0]*x) / w[1]
	plt.scatter(x,ybc,c='black', label='Naive Decision surface with C=sigma^2*I')

	w = np.matmul(np.linalg.inv(C_naive_diag), (mu1 - mu3))
	pc1 = 0.3
	pc3 = 0.3
	x0 = (mu1 + mu3) / 2 - np.log(pc1/pc3) * (mu1 - mu3) / np.matmul(np.matmul((mu1 - mu3).T, np.linalg.inv(C_naive_diag)), (mu1 - mu3))
	ybc = (np.matmul(w.T, x0) - w[0]*x) / w[1]
	plt.scatter(x,ybc,c='brown', label='Naive Decision surface for Red and Green')

	plt.xlabel("x")
	plt.ylabel("y")
	plt.legend(loc='upper left')
	plt.show()

def plotNaiveSameCovariance(x, X, Y, val, rv1, rv2, rv3, mu1, mu2, mu3, cov1, cov2, cov3):
	axes = plt.gca()
	axes.set_xlim([-10,50])
	axes.set_ylim([-10,50])

	c1 = plt.contour(X,Y,rv1.pdf(val), colors='r')
	plt.clabel(c1, fmt = '%0.6f')
	c2 = plt.contour(X,Y,rv2.pdf(val), colors='b')
	plt.clabel(c2, fmt = '%0.6f')
	c3 = plt.contour(X,Y,rv3.pdf(val), colors='g')
	plt.clabel(c3, fmt = '%0.6f')

	C = (cov1 + cov2 + cov3) / 3 * np.eye(cov1.shape[0])

	w = np.matmul(np.linalg.inv(C), (mu1 - mu2))
	pc1 = 0.3
	pc2 = 0.3
	x0 = (mu1 + mu2) / 2 - np.log(pc1/pc2) * (mu1 - mu2) / np.matmul(np.matmul((mu1 - mu2).T, np.linalg.inv(C)), (mu1 - mu2))
	ybc = (np.matmul(w.T, x0) - w[0]*x) / w[1]
	plt.scatter(x,ybc,c='black', label='Naive Decision surface for Red and Blue with same C')

	w = np.matmul(np.linalg.inv(C), (mu2 - mu3))
	pc2 = 0.3
	pc3 = 0.3
	x0 = (mu2 + mu3) / 2 - np.log(pc2/pc3) * (mu2 - mu3) / np.matmul(np.matmul((mu2 - mu3).T, np.linalg.inv(C)), (mu2 - mu3))
	ybc = (np.matmul(w.T, x0) - w[0]*x) / w[1]
	plt.scatter(x,ybc,c='blue', label='Naive Decision surface with same C')

	w = np.matmul(np.linalg.inv(C), (mu1 - mu3))
	pc1 = 0.3
	pc3 = 0.3
	x0 = (mu1 + mu3) / 2 - np.log(pc1/pc3) * (mu1 - mu3) / np.matmul(np.matmul((mu1 - mu3).T, np.linalg.inv(C)), (mu1 - mu3))
	ybc = (np.matmul(w.T, x0) - w[0]*x) / w[1]
	plt.scatter(x,ybc,c='brown', label='Naive Decision surface for Red and Green with same C')

	plt.xlabel("x")
	plt.ylabel("y")
	plt.legend(loc='upper left')
	plt.show()

def plotNaiveDiffCovariance(x, X, Y, val, rv1, rv2, rv3, mu1, mu2, mu3, cov1, cov2, cov3):
	axes = plt.gca()
	axes.set_xlim([-10,50])
	axes.set_ylim([-10,50])

	c1 = plt.contour(X,Y,rv1.pdf(val), colors='r')
	plt.clabel(c1, fmt = '%0.6f')
	c2 = plt.contour(X,Y,rv2.pdf(val), colors='b')
	plt.clabel(c2, fmt = '%0.6f')
	c3 = plt.contour(X,Y,rv3.pdf(val), colors='g')
	plt.clabel(c3, fmt = '%0.6f')

	pc1 = 0.3
	pc2 = 0.3
	pc3 = 0.3

	cov1 = cov1 * np.eye(cov1.shape[0])
	cov2 = cov2 * np.eye(cov2.shape[0])
	cov3 = cov3 * np.eye(cov3.shape[0])

	c1_inv = np.linalg.inv(cov1)
	c2_inv = np.linalg.inv(cov2)
	c3_inv = np.linalg.inv(cov3)

	W1 = -0.5 * c1_inv
	W2 = -0.5 * c2_inv
	W3 = -0.5 * c3_inv

	w1 = np.matmul(c1_inv, mu1)
	w2 = np.matmul(c2_inv, mu2)
	w3 = np.matmul(c3_inv, mu3)

	w01 = -0.5 * np.matmul(np.matmul(mu1.T, c1_inv), mu1) - 0.5 * np.log(np.linalg.det(cov1)) + np.log(pc1)
	w02 = -0.5 * np.matmul(np.matmul(mu2.T, c2_inv), mu2) - 0.5 * np.log(np.linalg.det(cov2)) + np.log(pc2)
	w03 = -0.5 * np.matmul(np.matmul(mu3.T, c3_inv), mu3) - 0.5 * np.log(np.linalg.det(cov3)) + np.log(pc3)
		
	a = W2[1][1] - W1[1][1]
	b1 = w2[1][0] - w1[1][0]
	b2 = W2[1][0] + W2[0][1] - W1[1][0] - W1[0][1]
	c1 = W2[0][0] - W1[0][0]
	c2 = w2[0][0] - w1[0][0]
	c3 = w02 - w01
	x_res = []
	y_res = []
	for p in x:
		b = b1 + p * b2
		c = p * p * c1 + p * c2 + c3 
		x_res += [p, p]
		y_res += np.roots([a,b,c]).tolist()
	plt.scatter(x_res, y_res, c='blue', label='Naive Decision surface for Red and Blue ')	
		
	a = W3[1][1] - W2[1][1]
	b1 = w3[1][0] - w2[1][0]
	b2 = W3[1][0] + W3[0][1] - W2[1][0] - W2[0][1]
	c1 = W3[0][0] - W2[0][0]
	c2 = w3[0][0] - w2[0][0]
	c3 = w03 - w02
	x_res = []
	y_res = []

	for p in x:
		b = b1 + p * b2
		c = p * p * c1 + p * c2 + c3 
		x_res += [p, p]
		y_res += np.roots([a,b,c]).tolist()
	plt.scatter(x_res, y_res, c='black', label='Naive Decision surface for Blue and Green')
		
	a = W1[1][1] - W3[1][1]
	b1 = w1[1][0] - w3[1][0]
	b2 = W1[1][0] + W1[0][1] - W3[1][0] - W3[0][1]
	c1 = W1[0][0] - W3[0][0]
	c2 = w1[0][0] - w3[0][0]
	c3 = w01 - w03
	x_res = []
	y_res = []
	for p in x:
		b = b1 + p * b2
		c = p * p * c1 + p * c2 + c3 
		x_res += [p, p]
		y_res += np.roots([a,b,c]).tolist()
	plt.scatter(x_res, y_res, c='orange', label='Naive Decision surface for Red and Green')

	plt.xlabel("x")
	plt.ylabel("y")
	plt.legend(loc='upper left')
	plt.show()

def plotContoursDecision(mu1, cov1, mu2, cov2, mu3, cov3):
	rv1 = multivariate_normal(mu1.T[0],cov1)
	rv2 = multivariate_normal(mu2.T[0],cov2)
	rv3 = multivariate_normal(mu3.T[0],cov3)

	x = np.linspace(-10,50,1000)
	y = np.linspace(-10,50,1000)

	#X <- x replicated y times row wise, Y <- y replicated x times col wise
	X, Y = np.meshgrid(x,y)
	val = np.empty(X.shape + (2,))
	val[:, :, 0] = X 
	val[:, :, 1] = Y

	plotSameBayesCovariance(x, X,Y, val, rv1, rv2, rv3, mu1, mu2, mu3, cov1, cov2, cov3)
	plotDiffBayesCovariance(x, X,Y, val, rv1, rv2, rv3, mu1, mu2, mu3, cov1, cov2, cov3)
	plotNaiveSigma2(x, X,Y, val, rv1, rv2, rv3, mu1, mu2, mu3, cov1, cov2, cov3)
	plotNaiveSameCovariance(x, X,Y, val, rv1, rv2, rv3, mu1, mu2, mu3, cov1, cov2, cov3)
	plotNaiveDiffCovariance(x, X,Y, val, rv1, rv2, rv3, mu1, mu2, mu3, cov1, cov2, cov3)

def plotGaussian(mu1, cov1, mu2, cov2, mu3, cov3):
	rv1 = multivariate_normal(mu1,cov1)
	rv2 = multivariate_normal(mu2,cov2)
	rv3 = multivariate_normal(mu3,cov3)

	x = np.linspace(0,3000,1000)
	y = np.linspace(0,3000,1000)

	#X <- x replicated y times row wise, Y <- y replicated x times col wise
	X, Y = np.meshgrid(x,y)
	val = np.empty(X.shape + (2,))
	val[:, :, 0] = X 
	val[:, :, 1] = Y

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	
	surf1 = ax.plot_surface(X, Y, rv1.pdf(val), color='red')
	surf2 = ax.plot_surface(X, Y, rv2.pdf(val), color='blue')
	surf3 = ax.plot_surface(X, Y, rv3.pdf(val), color='green')
	
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

	plt.show()

#############################################Calculating functions#################################################################	
def calcMuSigma(data):
	data1 = data[np.equal(data[:,-1],1), :]
	data2 = data[np.equal(data[:,-1],2), :]
	data3 = data[np.equal(data[:,-1],3), :]

	m1 = data1.shape[0]
	m2 = data2.shape[0]
	m3 = data3.shape[0]
	
	#mu are column vectors
	mu1 = data1[:,:-1].sum(axis=0, keepdims=True).T / m1
	mu2 = data2[:,:-1].sum(axis=0, keepdims=True).T / m2
	mu3 = data3[:,:-1].sum(axis=0, keepdims=True).T / m3

	y1 = data1[:,:-1] - mu1.T
	y2 = data2[:,:-1] - mu2.T
	y3 = data3[:,:-1] - mu3.T

	cov1 = 1/(m1-1) * np.matmul(y1.T, y1)
	cov2 = 1/(m2-1) * np.matmul(y2.T, y2)
	cov3 = 1/(m3-1) * np.matmul(y3.T, y3)

	return mu1, mu2, mu3, cov1, cov2, cov3

def splitData(data, train_ratio=0.7, val_ratio=0.2):
	data_size = data.shape[0]
	train_size = (int)(data_size * train_ratio)
	val_size = (int)(data_size * val_ratio)
	test_size = data_size - train_size - val_size

	train_data = data[0:train_size,:]
	val_data = data[train_size:train_size+val_size,:]
	test_data = data[train_size+val_size:,:]

	return train_data, val_data, test_data

def plotROC(fp1, tp1, fp2, tp2, fp3, tp3, lab):
	x1 = np.arange(min(fp1),max(fp1),0.01)
	f1 = interp1d(fp1, tp1)
	y1 = f1(x1)
	plt.plot(fp1,tp1,'o')
	plt.plot(x1,y1,'-',label='ROC Class 1 - '+lab)

	x2 = np.arange(min(fp2),max(fp2),0.01)
	f2 = interp1d(fp2, tp2)
	y2 = f2(x2)
	plt.plot(fp2,tp2,'o')
	plt.plot(x2,y2,'-',label='ROC Class 2 - '+lab)
	
	x3 = np.arange(min(fp3),max(fp3),0.01)
	f3 = interp1d(fp3, tp3)
	y3 = f3(x3)
	plt.plot(fp3,tp3,'o')
	plt.plot(x3,y3,'-',label='ROC Class 3 - '+lab)
	
	x2_inter = [i if min(fp2) <= i <= max(fp2) else (min(fp2) if i < min(fp2) else max(fp2)) for i in x1]
	x3_inter = [i if min(fp3) <= i <= max(fp3) else (min(fp3) if i < min(fp3) else max(fp3)) for i in x1]
	
	y2_inter = f2(x2_inter)
	y3_inter = f3(x3_inter)

	y_avg = (y1 + y2_inter + y3_inter) / 3
	plt.plot(x1,y_avg,'-',label=lab)

	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.legend(loc='lower right')
	plt.show()

def plotDET(fp1, mr1, fp2, mr2, fp3, mr3, lab):
	probit_fp1 = norm.ppf(fp1)
	probit_mr1 = norm.ppf(mr1)
	
	temp = [(i,j) for i,j in zip(probit_fp1,probit_mr1) if not np.isinf(i) and not np.isinf(j)]
	probit_fp1 = np.array([i[0] for i in temp])
	probit_mr1 = np.array([i[1] for i in temp])
	
	x1 = np.arange(min(probit_fp1),max(probit_fp1),0.01)
	f1 = interp1d(probit_fp1, probit_mr1)
	y1 = f1(x1)
	plt.plot(probit_fp1, probit_mr1, 'o')
	plt.plot(x1,y1,'-',label='DET Class 1'+lab)

	probit_fp2 = norm.ppf(fp2)
	probit_mr2 = norm.ppf(mr2)
	
	temp = [(i,j) for i,j in zip(probit_fp2,probit_mr2) if not np.isinf(i) and not np.isinf(j)]
	probit_fp2 = np.array([i[0] for i in temp])
	probit_mr2 = np.array([i[1] for i in temp])
	
	x2 = np.arange(min(probit_fp2),max(probit_fp2),0.01)
	f2 = interp1d(probit_fp2, probit_mr2)
	y2 = f2(x2)
	plt.plot(probit_fp2, probit_mr2, 'o')
	plt.plot(x2,y2,'-',label='DET Class 2'+lab)
	
	probit_fp3 = norm.ppf(fp3)
	probit_mr3 = norm.ppf(mr3)
	
	temp = [(i,j) for i,j in zip(probit_fp3,probit_mr3) if not np.isinf(i) and not np.isinf(j)]
	probit_fp3 = np.array([i[0] for i in temp])
	probit_mr3 = np.array([i[1] for i in temp])
	
	x3 = np.arange(min(probit_fp3),max(probit_fp3),0.01)
	f3 = interp1d(probit_fp3, probit_mr3)
	y3 = f3(x3)
	plt.plot(probit_fp3, probit_mr3, 'o')
	plt.plot(x3,y3,'-',label='DET Class 3'+lab)
	
	x2_inter = [i if min(probit_fp2) <= i <= max(probit_fp2) else (min(probit_fp2) if i < min(probit_fp2) else max(probit_fp2)) for i in x1]
	x3_inter = [i if min(probit_fp3) <= i <= max(probit_fp3) else (min(probit_fp3) if i < min(probit_fp3) else max(probit_fp3)) for i in x1]
	
	y2_inter = f2(x2_inter)
	y3_inter = f3(x3_inter)

	y_avg = (y1 + y2_inter + y3_inter) / 3
	plt.plot(x1,y_avg,'-',label=lab)

	plt.xlabel('False Positive Rate')
	plt.ylabel('Miss Rate')
	plt.legend(loc='lower right')
	plt.show()

def plotROCDET(val_data, mu1, mu2, mu3, cov1, cov2, cov3):
	#Full covariance Bayesian classifier on Non Linear Separable data
	tp1, fp1, mr1 = [], [], []
	tp2, fp2, mr2 = [], [], []
	tp3, fp3, mr3 = [], [], []
	
	for thr1 in np.arange(-4,4,0.05):
		conf_matrix,avg_acc,tpr1,tpr2,tpr3,fpr1,fpr2,fpr3,m1,m2,m3 = confusion_matrix(val_data, mu1, mu2, mu3, cov1, cov2, cov3, thr1, 0, 0)	
		tp1 += [tpr1]
		fp1 += [fpr1]
		mr1 += [m1]

	for thr2 in np.arange(-4,4,0.05):
		conf_matrix,avg_acc,tpr1,tpr2,tpr3,fpr1,fpr2,fpr3,m1,m2,m3 = confusion_matrix(val_data, mu1, mu2, mu3, cov1, cov2, cov3, 0, thr2, 0)	
		tp2 += [tpr2]
		fp2 += [fpr2]
		mr2 += [m2]
	
	for thr3 in np.arange(-4,4,0.05):
		conf_matrix,avg_acc,tpr1,tpr2,tpr3,fpr1,fpr2,fpr3,m1,m2,m3 = confusion_matrix(val_data, mu1, mu2, mu3, cov1, cov2, cov3, 0, 0, thr3)	
		tp3 += [tpr3]
		fp3 += [fpr3]
		mr3 += [m3]
	
	plotROC(fp1, tp1, fp2, tp2, fp3, tp3, 'Bayesian Full Cov')
	plotDET(fp1, mr1, fp2, mr2, fp3, mr3, 'Bayesian Full Cov')

	#Same covariance Bayesian classifier on Non Linear Separable data
	cov = (cov1 + cov2 + cov3) / 3
	tp1, fp1, mr1 = [], [], []
	tp2, fp2, mr2 = [], [], []
	tp3, fp3, mr3 = [], [], []
	
	for thr1 in np.arange(-4,4,0.05):
		conf_matrix,avg_acc,tpr1,tpr2,tpr3,fpr1,fpr2,fpr3,m1,m2,m3 = confusion_matrix(val_data, mu1, mu2, mu3, cov, cov, cov, thr1, 0, 0)	
		tp1 += [tpr1]
		fp1 += [fpr1]
		mr1 += [m1]
	
	for thr2 in np.arange(-4,4,0.05):
		conf_matrix,avg_acc,tpr1,tpr2,tpr3,fpr1,fpr2,fpr3,m1,m2,m3 = confusion_matrix(val_data, mu1, mu2, mu3, cov, cov, cov, 0, thr2, 0)	
		tp2 += [tpr2]
		fp2 += [fpr2]
		mr2 += [m2]
	
	for thr3 in np.arange(-4,4,0.05):
		conf_matrix,avg_acc,tpr1,tpr2,tpr3,fpr1,fpr2,fpr3,m1,m2,m3 = confusion_matrix(val_data, mu1, mu2, mu3, cov, cov, cov, 0, 0, thr3)	
		tp3 += [tpr3]
		fp3 += [fpr3]
		mr3 += [m3]

	plotROC(fp1, tp1, fp2, tp2, fp3, tp3, 'Bayesian Same Cov')
	plotDET(fp1, mr1, fp2, mr2, fp3, mr3, 'Bayesian Same Cov')

#####################################################Run Experiment##################################################################
def runExperiment(data):
	np.random.shuffle(data)

	train_data, val_data, test_data = splitData(data)

	mu1, mu2, mu3, cov1, cov2, cov3 = calcMuSigma(train_data)
	
	plotContoursDecision(mu1, cov1, mu2, cov2, mu3, cov3)
	plotGaussian(mu1, cov1, mu2, cov2, mu3, cov3)
	plotROCDET(val_data, mu1, mu2, mu3, cov1, cov2, cov3)
	
######################################################Argument Parser#################################################################				
ap=argparse.ArgumentParser()

ap.add_argument('-ls','--ls',required=True,help='Enter the path to the Linearly separable data file')
ap.add_argument('-nls','--nls',required=True,help='Enter the path to the Non Linearly separable data file')
ap.add_argument('-rd','--rd',required=True,help='Enter the path to the Real data file')

args=vars(ap.parse_args())
#############################################Helper functions######################################################################			
if __name__ == '__main__':
	#Linearly separable, Real, Non Linearly Separable data
	ls_data = getDataFromFile(args['ls'])
	r_data = getDataFromFile(args['rd'])
	nls_data = getDataFromFile(args['nls'])
	
	#Set the axes of plots depending upon type of data
	runExperiment(ls_data)
	runExperiment(r_data)
	runExperiment(nls_data)
