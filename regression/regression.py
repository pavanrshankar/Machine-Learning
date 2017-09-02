import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import sys
###################################Plot functions#######################################
def plotDegree(x_axis, y_axis, titl, clr, mar, marker_size):
	plt.scatter(x_axis, y_axis[:,[0]], color=clr, marker=mar, s=marker_size)
	plt.xlabel('Degree chosen')
	plt.ylabel('Number of iterations')
	plt.title(titl)
	plt.show()
	
def plotAverageError(avg_train_error, avg_val_error, polynomial_degree, set_axis=False):
	if set_axis == True:
		axes = plt.gca()
		axes.set_xlim([0,10])
		axes.set_ylim([0,100])
	
	plt.plot(polynomial_degree, avg_train_error[:,0:1], color='red', label='avg train error')
	plt.plot(polynomial_degree, avg_val_error[:,0:1], color='blue', label='avg validation error')
	plt.legend(loc='upper left')
	plt.xlabel('Degree of Polynomial')
	plt.ylabel('Average Error')
	plt.show()

def plot3DCurve(train_data, pred_y):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.scatter(train_data[:,0:1], train_data[:,1:2], train_data[:,2:3], c='red', label='train_data')
	ax.scatter(train_data[:,0:1], train_data[:,1:2], pred_y, c='blue', label='predicted_data')
	plt.legend(loc='upper left')
	plt.show()

def plotRegularized3DCurve(train_data, pred_y, pred_y_l):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.scatter(train_data[:,0:1], train_data[:,1:2], train_data[:,2:3], c='red', label='train_data')
	ax.scatter(train_data[:,0:1], train_data[:,1:2], pred_y, c='green', label='predicted_data')
	ax.scatter(train_data[:,0:1], train_data[:,1:2], pred_y_l, c='purple', label='predicted_data_with_ridge_regularization')
	plt.legend(loc='upper left')
	plt.show()

def plotCurve(train_data, pred_y):
	plt.scatter(train_data[:,0:1], train_data[:,1:2], color='red', label='train_data')
	plt.scatter(train_data[:,0:1], pred_y, color='green', label='predicted_data')
	plt.legend(loc='lower left')
	plt.show()

def plotRegularizedCurves(train_data, pred_y, pred_y_l):
	plt.scatter(train_data[:,0:1], train_data[:,1:2], color='red', label='train_data')
	plt.scatter(train_data[:,0:1], pred_y, color='green', label='predicted_data')
	plt.scatter(train_data[:,0:1], pred_y_l, color='blue', label='predicted_data_with_ridge_regularization')
	plt.legend(loc='lower left')
	plt.title(r'$\lambda = 0.0005$')
	plt.show()	

def plotAvgErrorLambda(x, avg_train, avg_val):
	axes = plt.gca()
	plt.scatter(x ,avg_train, color='red', label='avg_train_error')
	plt.scatter(x ,avg_val, color='blue', label='avg_val_error')	
	plt.xlabel(r'log($\lambda$)')
	plt.ylabel('Average Error')
	plt.legend(loc='upper left')
	plt.show()	

def plotData(data, dim):
	if dim == 1:
		plt.scatter(data[:,0:1], data[:,1:2], color='red')
		plt.xlabel('x')
		plt.ylabel('y')
		plt.show()
	else:
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		ax.scatter(data[:,0:1], data[:,1:2], data[:,2:3], color='red')
		ax.set_xlabel('x1')
		ax.set_ylabel('x2')
		ax.set_zlabel('y')
		plt.show()	

def plotRegularized3DSurface(train_data, pred_y, pred_y_l):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	x = train_data[:,0:1].flatten()
	y = train_data[:,1:2].flatten()
	z = train_data[:,2:3].flatten()
	p = pred_y.flatten()
	p_l = pred_y_l.flatten()
	ax.plot_trisurf(x, y, z, color='red')
	ax.plot_trisurf(x, y, p, color='green')
	ax.plot_trisurf(x, y, p_l, color='blue')
	plt.show()

def plot3DSurface(train_data, pred_y):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	x = train_data[:,0:1].flatten()
	y = train_data[:,1:2].flatten()
	z = train_data[:,2:3].flatten()
	p = pred_y.flatten()
	ax.plot_trisurf(x, y, z, color='red')
	ax.plot_trisurf(x, y, p, color='blue')
	plt.show()
	
###################################Helper functions#######################################
def getRawData(path):
	df = pd.read_csv(path, header=None, delim_whitespace=True)
	data = np.array(df)
	return data

def getNextCrossValidatedData(data, kth_fold, k_fold_cross_validation, train_ratio=0.7, val_ratio=0.2):
	data_size = data.shape[0]
	train_size = (int)(data_size * train_ratio)
	val_size = (int)(data_size * val_ratio)
	test_size = data_size - train_size - val_size

	bucket_size = (int)(data_size / k_fold_cross_validation)

	#Circular data assignment
	start_train_ind = kth_fold * bucket_size

	#Training data
	if(start_train_ind + train_size <= data_size):
		train_data = data[start_train_ind:start_train_ind+train_size,:]
		start_val_ind = start_train_ind+train_size
	else:
		train_data = np.concatenate((data[start_train_ind:data_size,:], data[0:train_size-data_size+start_train_ind,:]), axis=0)
		start_val_ind = train_size-data_size+start_train_ind

	#Validation data	
	if(start_val_ind + val_size <= data_size):
		val_data = data[start_val_ind:start_val_ind+val_size,:]

		start_test_ind = start_val_ind+val_size
	else:
		val_data = np.concatenate((data[start_val_ind:data_size,:], data[0:val_size-data_size+start_val_ind,:]), axis=0)
		start_test_ind = val_size-data_size+start_val_ind
			

	#Testing data
	if(start_test_ind + test_size <= data_size):
		test_data = data[start_test_ind:start_test_ind+test_size,:]
		start_test_ind = start_test_ind+test_size
	else:
		test_data = np.concatenate((data[start_test_ind:data_size,:], data[0:test_size-data_size+start_test_ind,:]), axis=0)

	return train_data, val_data, test_data

def getX_Y(data, nd_regression):
	x = data[:,:-1]
	y = data[:,-1:]
	return x, y

def getError(coeff, true_x, true_y):
	pred_y = np.matmul(true_x, coeff)
	totalError = np.matmul(np.transpose(pred_y-true_y), pred_y-true_y)[0][0]
	return totalError, pred_y

def updateFeatures(x):
	new_feature = np.multiply(x[:,-1:], x[:,1:2])
	x = np.append(x, new_feature[:,0:1],1)
	return x
###################################Parameter Calculation functions#######################################
def calcParam(train_x, train_y, lamb):
	trans_x = np.transpose(train_x)
	num_feat = trans_x.shape[0]
	coeff = np.matmul(np.matmul(np.linalg.pinv(np.matmul(trans_x, train_x) + lamb*np.identity(num_feat)), trans_x), train_y)  
	return coeff

###################################Feature Manipulation functions#######################################
'''
	Algorithm used is similar to monomial generation for a degree n with k terms	
	Step wise illustration for degree 3 with 2 terms:

	multi = [x1, x2]
	curr_deg = [[x1], [x2]] 
	curr_deg = [[x1^2, x1.x2], [x2^2]]
	curr_deg = [[x1^3, x1^2.x2, x1.x2^2], [x2^3]]
	At each step flatten curr_deg and add it to train_x
'''
def generateInteractionFeatures(n, train_multi, val_multi, train_curr_deg, val_curr_deg):
	train_lat_deg = []
	val_lat_deg = []
			
	for i in range(n-1):
		train_res = []
		val_res = []

		for j in range(i, n-1):
			for ele in train_curr_deg[j]:
				train_res.append(np.multiply(train_multi[i],ele))
			for ele in val_curr_deg[j]:	
				val_res.append(np.multiply(val_multi[i],ele))

		train_lat_deg += [train_res]
		val_lat_deg += [val_res]

	return train_lat_deg, val_lat_deg

def standardizeFeatures(A):
	A = (A - np.mean(A, axis=0)) / np.std(A, axis=0)
	return A

def removeCorrelatedFeatures(data):
	x = data[:,:-1]
	corrMat = pd.DataFrame(x).corr()
	feat = corrMat.shape[1]
	
	updated_x = x[:,0:1]
	feat_sel = [0]

	for i in range(1, feat):
		col_corr = False
		for j in range(i):
			if(abs(corrMat.iloc[i,j]) >= 0.7 and j in feat_sel):
				col_corr = True
				break
		if(col_corr == False):
			feat_sel += [i]
			updated_x = np.concatenate((updated_x, x[:,i:i+1]),axis=1)

	return updated_x
###################################Regression function#######################################
def performRegression(train_ratio, val_ratio, max_degree, data, nd_regression, lamb):
	#Additional target column provided
	if(nd_regression > 2):
		data = data[:,:-1]

	#Remove correlated features
	clean_data = removeCorrelatedFeatures(data)
	clean_data = np.concatenate((clean_data, data[:,-1:]),axis=1)
	
	#Repeating experiment max_iter times by shuffling data
	max_iter = 1
	k_fold_cross_validation = 1

	avg_train_error = np.zeros((max_degree,1))
	avg_val_error = np.zeros((max_degree,1))

	for exper in range(max_iter):
		np.random.shuffle(clean_data)

		#Cross Validation for training to happen across entire data set
		for kth_fold in range(k_fold_cross_validation):
			train_data, val_data, test_data = getNextCrossValidatedData(clean_data, kth_fold, k_fold_cross_validation, train_ratio,val_ratio)

			train_x, train_y = getX_Y(train_data, nd_regression)
			val_x, val_y = getX_Y(val_data, nd_regression)	

			train_x = standardizeFeatures(train_x)
			val_x = standardizeFeatures(val_x)

			train_x = np.append(np.ones((len(train_x),1)), train_x ,1)	
			val_x = np.append(np.ones((len(val_x),1)), val_x ,1)
		
			if(nd_regression == 1):
				#Varying degree from 1 to max_degree
				for deg in range(1, max_degree+1):	
					if deg != 1:
						train_x = updateFeatures(train_x)
						val_x = updateFeatures(val_x)

					coeff_l = calcParam(train_x, train_y, lamb)
					trainError, pred_y_l = getError(coeff_l, train_x, train_y)

					coeff = calcParam(train_x, train_y, 0)
					trainError, pred_y = getError(coeff, train_x, train_y)
					
					valError, pred_y = getError(coeff_l, val_x, val_y)

					plotCurve(train_data, pred_y_l)
					#plotRegularizedCurves(train_data, pred_y, pred_y_l)

					avg_train_error[deg-1] += (trainError / train_ratio)
					avg_val_error[deg-1] += (valError / val_ratio)

			else:
				m = train_x.shape[0]
				n = train_x.shape[1]

				train_col_list = train_x[:,1:].T
				train_multi = [np.array(i) for i in train_col_list]
				train_curr_deg = [[np.array(i)] for i in train_col_list]

				val_col_list = val_x[:,1:].T
				val_multi = [np.array(i) for i in val_col_list]
				val_curr_deg = [[np.array(i)] for i in val_col_list]

				#Varying degree from 1 to max_degree:	
				for deg in range(1, max_degree+1):
					if deg != 1:
						train_lat_deg, val_lat_deg = generateInteractionFeatures(n, train_multi, val_multi, train_curr_deg, val_curr_deg)						

						new_feat = [monomial.tolist() for sublist in train_lat_deg for monomial in sublist]
						train_x = np.concatenate((train_x, np.array(new_feat).T), axis=1)

						new_feat = [monomial.tolist() for sublist in val_lat_deg for monomial in sublist]
						val_x = np.concatenate((val_x, np.array(new_feat).T), axis=1)

						train_curr_deg = train_lat_deg		
						val_curr_deg = val_lat_deg

					coeff_l = calcParam(train_x, train_y, lamb)
					trainError, pred_y_l = getError(coeff_l, train_x, train_y)

					coeff = calcParam(train_x, train_y, 0)
					trainError, pred_y = getError(coeff, train_x, train_y)

					valError, pred_y = getError(coeff_l, val_x, val_y)	
					
					#plot3DCurve(train_data, pred_y_l)
					#plot3DSurface(train_data, pred_y_l)
					#plotRegularized3DCurve(train_data, pred_y, pred_y_l)
					#plotRegularized3DSurface(train_data, pred_y, pred_y_l)

					avg_train_error[deg-1] += (trainError / train_ratio)
					avg_val_error[deg-1] += (valError / val_ratio)

		avg_train_error /= k_fold_cross_validation
		avg_val_error /= k_fold_cross_validation			
	
	avg_train_error /= max_iter
	avg_val_error /= max_iter

	#plotAverageError(avg_train_error, avg_val_error, np.arange(max_degree)+1)	
	return avg_train_error, avg_val_error
###################################Driver functions#######################################
#Change experiment parameters as required
def runExperiment(path, expno):
	data = getRawData(path)
	nd_regression = data.shape[1] - 1

	#Model Selection
	if expno == '1':
		train_ratio = 0.7
		val_ratio = 0.2
		max_degree = 6
		experiment_runs = 50
		lamb = 0
		
		deg_chosen_count = np.zeros((max_degree, 1))

		for exper in range(experiment_runs):
			avg_train_error, avg_val_error = performRegression(train_ratio, val_ratio, max_degree, data, nd_regression, lamb)
			min_val_error_degree = np.argmin(avg_val_error, axis=0)[0] + 1
			deg_chosen_count[min_val_error_degree-1] += 1
		#plotDegree(np.arange(max_degree)+1, deg_chosen_count, 'Number of times Degree is chosen out of 50 experiments', 'magenta', '+', 200)
	
	#Choosing lambda for regularization
	if expno == '2':
		train_ratio = 0.7
		val_ratio = 0.2
		max_degree = 6
		deg_cons = 4
		x = np.arange(-35, 10, 1)

		avg_train = []
		avg_val = []

		for lamb in x:
			avg_train_error, avg_val_error = performRegression(train_ratio, val_ratio, max_degree, data, nd_regression, math.exp(lamb))
			avg_train += avg_train_error[deg_cons-1].tolist()
			avg_val += avg_val_error[deg_cons-1].tolist()
		#plotAvgErrorLambda(x, avg_train, avg_val)	

#############################################Run###########################################	
if __name__ == '__main__':
	#Provide the data file and experiment number in command line
	filename = sys.argv[1]
	expno = sys.argv[2]
	runExperiment(filename, expno)
