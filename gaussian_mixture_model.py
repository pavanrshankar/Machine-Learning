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
from scipy.interpolate import interp1d

def normalize(A):
	A = (A - np.mean(A, axis=0)) / np.std(A, axis=0)
	return A

def getData(path):
	df = pd.read_csv(path, header=None, delim_whitespace=True)
	data = np.array(df)
	return data

#[pi1*N(template 1), pi2*N(template 2), .....]
def findTemplateDist(templates, pi, cov, mu, data):
	m,n = data.shape
	tempDist = []
	
	for i in range(templates):
		detcov = np.linalg.det(cov[i])	
		covinv = np.linalg.inv(cov[i])
		A = data - mu[i]
		mat = np.array([np.matmul(np.matmul(A[i,:], covinv), A.T[:, i]) for i in range(m)]).reshape((m,1))
		tempDist += [(pi[i] * 1/np.sqrt(pow(2*np.pi, n) * detcov)) * np.exp(-0.5 * mat)]

	return tempDist	

def splitData(data, train_ratio=0.7, val_ratio=0.15):
	data_size = data.shape[0]
	train_size = (int)(data_size * train_ratio)
	val_size = (int)(data_size * val_ratio)
	test_size = data_size - train_size - val_size

	train_data = data[0:train_size,:]
	val_data = data[train_size:train_size+val_size,:]
	test_data = data[train_size+val_size:,:]

	return train_data, val_data, test_data

def plotConfROCDET(trueResultVal, normtempDistVal, normtempDistValDiag, predResultVal, predResultValDiag):
	conf = metrics.confusion_matrix(trueResultVal, predResultVal)
	confDiag = metrics.confusion_matrix(trueResultVal, predResultValDiag) 

	print(conf)
	print(confDiag)

	fpr1, tpr1, _ = metrics.roc_curve(trueResultVal, normtempDistVal[1])	
	fpr2, tpr2, _ = metrics.roc_curve(trueResultVal, normtempDistValDiag[1])
	
	axes = plt.gca()
	axes.set_axisbelow(True)
	axes.minorticks_on()
	axes.grid(which='major', linestyle='-', linewidth='0.5', color='red')	
	axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

	plt.plot(fpr1,tpr1, c='b', label='Non-Diagonal')	
	plt.plot(fpr2,tpr2, c='r', label='Diagonal')	
	plt.legend(loc='lower right')
	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.title('ROC K=10')
	plt.show()

	mr1 = (1-np.array(tpr1)).tolist()
	mr2 = (1-np.array(tpr2)).tolist()

	probit_fp1 = norm.ppf(fpr1)
	probit_mr1 = norm.ppf(mr1)

	temp = [(i,j) for i,j in zip(probit_fp1,probit_mr1) if not np.isinf(i) and not np.isinf(j)]
	probit_fp1 = np.array([i[0] for i in temp])
	probit_mr1 = np.array([i[1] for i in temp])

	x1 = np.arange(min(probit_fp1),max(probit_fp1),0.01)
	f1 = interp1d(probit_fp1, probit_mr1)
	y1 = f1(x1)

	probit_fp2 = norm.ppf(fpr2)
	probit_mr2 = norm.ppf(mr2)

	temp = [(i,j) for i,j in zip(probit_fp2,probit_mr2) if not np.isinf(i) and not np.isinf(j)]
	probit_fp2 = np.array([i[0] for i in temp])
	probit_mr2 = np.array([i[1] for i in temp])

	x2 = np.arange(min(probit_fp2),max(probit_fp2),0.01)
	f2 = interp1d(probit_fp2, probit_mr2)
	y2 = f2(x2)

	axes = plt.gca()
	axes.set_axisbelow(True)
	axes.minorticks_on()
	axes.grid(which='major', linestyle='-', linewidth='0.5', color='red')	
	axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

	plt.plot(x1,y1,'-',c='b', label='DET Non-Diagonal')
	plt.plot(x2,y2,'-',c='r', label='DET Diagonal')
	plt.legend(loc='lower left')
	plt.xlabel('Probit FPR')
	plt.ylabel('Probit MR')
	plt.title('DET K=10')
	plt.show()

def giveVal(x):
	if x > 0: 
		return 7
	else: 
		return -7

def plotConfROCDETRealData(trueResultVal, normtempDistVal, normtempDistValDiag, predResultValImg, predResultValDiagImg, k):
	#conf = metrics.confusion_matrix(trueResultVal, predResultValImg)
	confDiag = metrics.confusion_matrix(trueResultVal, predResultValDiagImg) 
	print(confDiag)

	tot_vec = normtempDistVal[0].shape[0]
	vec_img = 36
	tot_img = (int)(tot_vec / vec_img)

	normtempDistValDiagImg = []
	for clas in normtempDistValDiag:
		normtempDistValDiagImg += [np.mean(clas.reshape((tot_img, vec_img)), axis=1).T]
		
	trueResultVal1 = [1 if i == 0 else 0 for i in trueResultVal]
	fpr1, tpr1, _ = metrics.roc_curve(trueResultVal1, normtempDistValDiagImg[0])
	print(tpr1)	

	trueResultVal2 = [1 if i == 1 else 0 for i in trueResultVal]
	fpr2, tpr2, _ = metrics.roc_curve(trueResultVal2, normtempDistValDiagImg[1])	

	trueResultVal3 = [1 if i == 2 else 0 for i in trueResultVal]
	fpr3, tpr3, _ = metrics.roc_curve(trueResultVal3, normtempDistValDiagImg[2])	
	
	f1 = interp1d(fpr1, tpr1)
	f2 = interp1d(fpr2, tpr2)
	f3 = interp1d(fpr3, tpr3)

	range_x = np.arange(0,1,0.01)

	tpr1_inter = f1(range_x)
	tpr2_inter = f2(range_x)
	tpr3_inter = f3(range_x)

	tpr_avg = (tpr1_inter + tpr2_inter + tpr3_inter) / 3

	axes = plt.gca()
	axes.set_axisbelow(True)
	axes.minorticks_on()
	axes.grid(which='major', linestyle='-', linewidth='0.5', color='red')	
	axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
	plt.plot(range_x,tpr_avg, c='black', label='ROC Real Diaginal')	
	plt.legend(loc='lower right')
	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.title('K='+str(k))
	plt.show()
	
	mr1 = (1-np.array(tpr1)).tolist()
	mr2 = (1-np.array(tpr2)).tolist()
	mr3 = (1-np.array(tpr3)).tolist()

	probit_fpr1 = norm.ppf(fpr1)
	probit_mr1 = norm.ppf(mr1)

	temp = [(giveVal(i), giveVal(j)) if (np.isinf(i) and np.isinf(j)) else (giveVal(i), j) if (np.isinf(i)) else (i,giveVal(j)) if (np.isinf(j)) else (i,j) for i,j in zip(probit_fpr1,probit_mr1)]
	probit_fpr1 = np.array([i[0] for i in temp])
	probit_mr1 = np.array([i[1] for i in temp])

	x1 = np.arange(min(probit_fpr1),max(probit_fpr1),1)
	f1 = interp1d(probit_fpr1, probit_mr1)
	y1 = f1(x1)

	probit_fpr2 = norm.ppf(fpr2)
	probit_mr2 = norm.ppf(mr2)

	temp = [(giveVal(i), giveVal(j)) if (np.isinf(i) and np.isinf(j)) else (giveVal(i), j) if (np.isinf(i)) else (i,giveVal(j)) if (np.isinf(j)) else (i,j) for i,j in zip(probit_fpr2,probit_mr2)]
	probit_fpr2 = np.array([i[0] for i in temp])
	probit_mr2 = np.array([i[1] for i in temp])

	x2 = np.arange(min(probit_fpr2),max(probit_fpr2),1)
	f2 = interp1d(probit_fpr2, probit_mr2)
	y2 = f2(x2)

	probit_fpr3 = norm.ppf(fpr3)
	probit_mr3 = norm.ppf(mr3)

	temp = [(giveVal(i), giveVal(j)) if (np.isinf(i) and np.isinf(j)) else (giveVal(i), j) if (np.isinf(i)) else (i,giveVal(j)) if (np.isinf(j)) else (i,j) for i,j in zip(probit_fpr3,probit_mr3)]
	probit_fpr3 = np.array([i[0] for i in temp])
	probit_mr3 = np.array([i[1] for i in temp])

	x3 = np.arange(min(probit_fpr3),max(probit_fpr3),1)
	f3 = interp1d(probit_fpr3, probit_mr3)
	y3 = f3(x3)

	y_avg = (y1 + y2 + y3) / 3

	range_x = np.arange(-3,3,0.01)
	y_avg = (f1(range_x) + f2(range_x) + f3(range_x)) / 3

	axes = plt.gca()
	axes.set_axisbelow(True)
	axes.minorticks_on()
	axes.grid(which='major', linestyle='-', linewidth='0.5', color='red')	
	axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
	plt.plot(range_x,y_avg,'-',c='black', label='DET Diagonal')

	plt.legend(loc='lower left')
	plt.xlabel('Probit FPR')
	plt.ylabel('Probit MR')
	plt.title('DET K'+str(k))
	plt.show()
	
def buildGMM(trained_data, val_data, templates, em_steps):
	no_classes = len(trained_data)

	mu = [[] for i in range(no_classes)]
	cov = [[] for i in range(no_classes)]
	pi = [[] for i in range(no_classes)]
	
	#Initialization step in GMM by K-Means
	labels = []
	m = []

	for i in range(no_classes):
		labels += [KMeans(n_clusters=templates, random_state=0).fit_predict(trained_data[i])]
		m += [trained_data[i].shape[0]]

	for i in range(no_classes):
		for j in range(templates):
			mu[i] += [np.mean(trained_data[i][labels[i] == j], axis=0)]
			cov[i] += [np.cov(trained_data[i][labels[i] == j].T)]
			pi[i] += [trained_data[i][labels[i] == j].shape[0]/m[i]]
			
	#Expectation Maximization Algorithm
	for i in range(em_steps):
		tempDist = []
		gam = []
		for i in range(no_classes):
			tempDist += [findTemplateDist(templates, pi[i], cov[i], mu[i], trained_data[i])]
			#[Class 1: pi * N(template i) / total (pi * N(template i)), Class2: pi * N(template i) / total (pi * N(template i)), ....]
			gam += [tempDist[i]/sum(tempDist[i])]


		#Recalculation of Parameters
		mu = [[] for i in range(no_classes)]
		cov = [[] for i in range(no_classes)]
		pi = [[] for i in range(no_classes)]

		for j in range(no_classes):
			for i in range(templates):
				pi[j] += [sum(gam[j][i])[0] / m[j]]			
				mu[j] += [sum(gam[j][i].reshape((m[j],1)) * trained_data[j]) / sum(gam[j][i])]
				cov[j] += [1/sum(gam[j][i])[0] * np.matmul((trained_data[j] - mu[j][i]).T, (gam[j][i].reshape((m[j],1)) * (trained_data[j] - mu[j][i])))]

	#Likelihood calculations p(x|theta)
	train_tot = trained_data[0]
	val_tot = val_data[0]
	for i in range(1,no_classes):
		train_tot = np.concatenate((train_tot, trained_data[i]), axis=0)
		val_tot = np.concatenate((val_tot, val_data[i]), axis=0)

	#Diagonal Covariance
	cov_diag = [[] for i in range(no_classes)]
	for i in range(no_classes):
		for j in range(templates):
			cov_diag[i] += [np.diag(cov[i][j]) * np.eye(cov[i][j].shape[0])]
	
	sumtempDistTrain = []
	sumtempDistVal = []
	sumtempDistTrainDiag = []
	sumtempDistValDiag = []

	#findTemplateDist(templates, pi[1], cov[1], mu[1], train_tot, True)[3]
	for i in range(no_classes):
		sumtempDistTrain += [sum(findTemplateDist(templates, pi[i], cov[i], mu[i], train_tot))]
		sumtempDistVal += [sum(findTemplateDist(templates, pi[i], cov[i], mu[i], val_tot))]

		sumtempDistTrainDiag += [sum(findTemplateDist(templates, pi[i], cov_diag[i], mu[i], train_tot))]
		sumtempDistValDiag += [sum(findTemplateDist(templates, pi[i], cov_diag[i], mu[i], val_tot))]

	if(no_classes == 2):
		###########
		trueResultTrain = [0]*len(trained_data[0]) + [1]*len(trained_data[1])
		trueResultVal = [0]*len(val_data[0]) + [1]*len(val_data[1]) 

		predResultTrainInd = (sumtempDistTrain[0] >= sumtempDistTrain[1]).flatten()
		predResultTrain = [0 if i==True else 1 for i in predResultTrainInd]

		predResultTrainDiagInd = (sumtempDistTrainDiag[0] >= sumtempDistTrainDiag[1]).flatten()
		predResultTrainDiag = [0 if i==True else 1 for i in predResultTrainDiagInd]

		predResultValInd = (sumtempDistVal[0] >= sumtempDistVal[1]).flatten()
		predResultVal = [0 if i==True else 1 for i in predResultValInd]

		predResultValDiagInd = (sumtempDistValDiag[0] >= sumtempDistValDiag[1]).flatten()
		predResultValDiag = [0 if i==True else 1 for i in predResultValDiagInd]

		train_accuracy = len([i for i, j in zip(trueResultTrain, predResultTrain) if i == j]) / len(trueResultTrain)
		train_accuracyDiag = len([i for i, j in zip(trueResultTrain, predResultTrainDiag) if i == j]) / len(trueResultTrain)

		val_accuracy = len([i for i, j in zip(trueResultVal, predResultVal) if i == j]) / len(trueResultVal)
		val_accuracyDiag = len([i for i, j in zip(trueResultVal, predResultValDiag) if i == j]) / len(trueResultVal)

		############
		totalScore = sum(np.array(sumtempDistVal))
		normtempDistVal = [ele/totalScore for ele in sumtempDistVal]

		totalScore = sum(np.array(sumtempDistTrain))
		normtempDistTrain = [ele/totalScore for ele in sumtempDistTrain]

		totalScoreDiag = sum(np.array(sumtempDistValDiag))
		normtempDistValDiag = [ele/totalScoreDiag for ele in sumtempDistValDiag]

		totalScoreDiag = sum(np.array(sumtempDistTrainDiag))
		normtempDistTrainDiag = [ele/totalScoreDiag for ele in sumtempDistTrainDiag]

		############
		return train_accuracy, val_accuracy, train_accuracyDiag, val_accuracyDiag, train_tot[predResultTrainInd], train_tot[np.logical_not(predResultTrainInd)], train_tot[predResultTrainDiagInd], train_tot[np.logical_not(predResultTrainDiagInd)],mu, cov, cov_diag, trueResultVal, normtempDistVal, normtempDistValDiag, trueResultTrain, normtempDistTrain, normtempDistTrainDiag, predResultVal, predResultValDiag

	else:
		rows_per_image = 36
		
		train_imgperclass = []
		val_imgperclass = []
		for i in range(no_classes):
			train_imgperclass += [(int) (len(trained_data[i]) / rows_per_image)]
			val_imgperclass += [(int) (len(val_data[i]) / rows_per_image)]

		trueResultTrain = []
		trueResultVal = []	
		predResultTrain = []
		predResultTrainDiag = []
		predResultVal = []
		predResultValDiag = []
		for i in range(no_classes):
			trueResultTrain += [i]*train_imgperclass[i]
			trueResultVal += [i]*val_imgperclass[i]
		
			predResultTrain += [-1]*len(trained_data[i])
			predResultTrainDiag +=  [-1]*len(trained_data[i])
			predResultVal += [-1]*len(val_data[i])
			predResultValDiag += [-1]*len(val_data[i])

		#########################
		predResultTrain_Ind = []
		for i in range(no_classes):
			res = (sumtempDistTrain[i] >= sumtempDistTrain[0])
			for j in range(1, no_classes):
				res = np.logical_and(res, (sumtempDistTrain[i] >= sumtempDistTrain[j]))

			predResultTrain_Ind += [res]	

		for i in range(len(predResultTrain)):
			for j in range(no_classes):
				if predResultTrain_Ind[j][i] == True:
					predResultTrain[i] = j

		no_images = (int)(len(predResultTrain) / rows_per_image)
		predResultTrainImg = np.array(predResultTrain).reshape((no_images, rows_per_image))
		predResultTrainImg = mode(predResultTrainImg.T)[0][0].tolist()

		#########################
		predResultTrainDiag_Ind = []
		for i in range(no_classes):
			res = (sumtempDistTrainDiag[i] >= sumtempDistTrainDiag[0])
			for j in range(1, no_classes):
				res = np.logical_and(res, (sumtempDistTrainDiag[i] >= sumtempDistTrainDiag[j]))
			predResultTrainDiag_Ind += [res]	

		for i in range(len(predResultTrainDiag)):
			for j in range(no_classes):
				if predResultTrainDiag_Ind[j][i] == True:
					predResultTrainDiag[i] = j

		no_images = (int)(len(predResultTrainDiag) / rows_per_image)
		predResultTrainDiagImg = np.array(predResultTrainDiag).reshape((no_images, rows_per_image))
		predResultTrainDiagImg = mode(predResultTrainDiagImg.T)[0][0].tolist()	

		#########################
		predResultVal_Ind = []
		for i in range(no_classes):
			res = (sumtempDistVal[i] >= sumtempDistVal[0])
			for j in range(1, no_classes):
				res = np.logical_and(res, (sumtempDistVal[i] >= sumtempDistVal[j]))

			predResultVal_Ind += [res]	

		for i in range(len(predResultVal)):
			for j in range(no_classes):
				if predResultVal_Ind[j][i] == True:
					predResultVal[i] = j

		no_images = (int)(len(predResultVal) / rows_per_image)
		predResultValImg = np.array(predResultVal).reshape((no_images, rows_per_image))
		predResultValImg = mode(predResultValImg.T)[0][0].tolist()	
		
		#########################
		predResultValDiag_Ind = []
		for i in range(no_classes):
			res = (sumtempDistValDiag[i] >= sumtempDistValDiag[0])
			for j in range(1, no_classes):
				res = np.logical_and(res, (sumtempDistValDiag[i] >= sumtempDistValDiag[j]))
			predResultValDiag_Ind += [res]	

		for i in range(len(predResultValDiag)):
			for j in range(no_classes):
				if predResultValDiag_Ind[j][i] == True:
					predResultValDiag[i] = j

		no_images = (int)(len(predResultValDiag) / rows_per_image)
		predResultValDiagImg = np.array(predResultValDiag).reshape((no_images, rows_per_image))
		predResultValDiagImg = mode(predResultValDiagImg.T)[0][0].tolist()		

		#########################
		train_accuracy = len([i for i, j in zip(trueResultTrain, predResultTrainImg) if i == j]) / len(trueResultTrain)
		train_accuracyDiag = len([i for i, j in zip(trueResultTrain, predResultTrainDiagImg) if i == j]) / len(trueResultTrain)
		
		val_accuracy = len([i for i, j in zip(trueResultVal, predResultValImg) if i == j]) / len(trueResultVal)
		val_accuracyDiag = len([i for i, j in zip(trueResultVal, predResultValDiagImg) if i == j]) / len(trueResultVal)

		predResultTrainInd = []
		predResultTrainDiagInd = []
		predResultValInd = []
		predResultValDiagInd = []
		for j in range(no_classes):
			predResultTrainInd += [True if i==j else False for i in predResultTrain]
			predResultTrainDiagInd = [True if i==j else False for i in predResultTrainDiag]
			predResultValInd = [True if i==j else False for i in predResultVal]
			predResultValDiagInd = [True if i==j else False for i in predResultValDiag]

		#########################
		totalScore = sum(np.array(sumtempDistVal))
		normtempDistVal = [ele/totalScore for ele in sumtempDistVal]

		totalScoreDiag = sum(np.array(sumtempDistValDiag))
		normtempDistValDiag = [ele/totalScoreDiag for ele in sumtempDistValDiag]

		#########################

		return train_accuracy, val_accuracy, train_accuracyDiag, val_accuracyDiag, train_tot[predResultTrainInd[0]], train_tot[predResultTrainInd[1]], train_tot[predResultTrainInd[2]], train_tot[predResultTrainDiagInd[0]], train_tot[predResultTrainDiagInd[1]], train_tot[predResultTrainDiagInd[2]], mu, cov, cov_diag, pi, trueResultVal, normtempDistVal, normtempDistValDiag, predResultValImg, predResultValDiagImg	

def analyzeSynthesizedData():
	data1 = normalize(getData('/Users/pavan.ravishankar/Downloads/pr3ass/synthetic_data/class1.txt'))
	data2 = normalize(getData('/Users/pavan.ravishankar/Downloads/pr3ass/synthetic_data/class2.txt'))

	train_data1, val_data1, test_data1 = splitData(data1)
	train_data2, val_data2, test_data2 = splitData(data2)

	plt.scatter(data1[:,0],data1[:,1], c='r', label='Class1')
	plt.scatter(data2[:,0],data2[:,1], c='b', label='Class2')
	plt.legend(loc='Upper right')
	plt.show()
	
	#KNN -> Only convex clusters
	k=4
	
	data = np.concatenate((data1, data2), axis=0)
	labels = KMeans(n_clusters=k, random_state=0).fit_predict(data)
	
	for i in range(k):
		plt.scatter(data[labels == i][:,0], data[labels == i][:,1], c='C'+str(i))
	plt.show()
	
	#GMM
	#Accuracy vs No of Templates
	
	no_templates = 30
	em_iter = 3

	train_acc = []
	val_acc = []
	train_accDiag = []
	val_accDiag = []
	for i in range(no_templates):
		train_accuracy, val_accuracy, train_accuracyDiag, val_accuracyDiag, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = buildGMM([train_data1, train_data2], [val_data1, val_data2], i+1, em_iter)
		train_acc += [train_accuracy]
		val_acc += [val_accuracy]
		
		train_accDiag += [train_accuracyDiag]
		val_accDiag += [val_accuracyDiag]

	axes = plt.gca()
	axes.set_ylim([0.5,1])	
	axes.set_axisbelow(True)
	axes.minorticks_on()
	axes.grid(which='major', linestyle='-', linewidth='0.5', color='red')	
	axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
	
	plt.scatter(np.arange(no_templates)+1,train_acc, c='r', marker='+')
	plt.plot(np.arange(no_templates)+1,train_acc, label='Training Accuracy Non Diagonal')
	
	plt.scatter(np.arange(no_templates)+1,val_acc, c='b', marker='+')
	plt.plot(np.arange(no_templates)+1,val_acc, label='Validation Accuracy Non Diagonal')

	plt.scatter(np.arange(no_templates)+1,train_accDiag, c='g', marker='+')
	plt.plot(np.arange(no_templates)+1,train_accDiag, label='Training Accuracy Diagonal')

	plt.scatter(np.arange(no_templates)+1,val_accDiag, c='black', marker='+')
	plt.plot(np.arange(no_templates)+1,val_accDiag, label='Validation Accuracy Diagonal')

	plt.title('Accuracy vs No of Templates of each class')
	plt.legend(loc='lower right')
	plt.show()
	
	#Contour Plots for k templates Non Diagonal
	no_templates = 10
	em_iter = 3

	_, _, _, _, data1, data2, data1Diag, data2Diag, mu, cov, cov_diag, trueResultVal, normtempDistVal, normtempDistValDiag, trueResultTrain, normtempDistTrain, normtempDistTrainDiag, predResultVal, predResultValDiag = buildGMM([train_data1, train_data2], [val_data1, val_data2], no_templates, em_iter)
	
	plotConfROCDET(trueResultVal, normtempDistVal, normtempDistValDiag, predResultVal, predResultValDiag)
	
	x = np.linspace(-2,2,100)
	y = np.linspace(-1.5,2,100)
	X, Y = np.meshgrid(x,y)
	val = np.empty(X.shape + (2,))
	val[:, :, 0] = X 
	val[:, :, 1] = Y

	rv1 = []
	rv2 = []
	for i in range(no_templates):
		rv1 += [multivariate_normal(mu[0][i],cov[0][i])]
		rv2 += [multivariate_normal(mu[1][i],cov[1][i])]

	for i in range(no_templates):
		c1 = plt.contour(X,Y,rv1[i].pdf(val), colors='black')
		plt.clabel(c1, fmt = '%1.3f')

		c2 = plt.contour(X,Y,rv2[i].pdf(val), colors='magenta')
		plt.clabel(c2, fmt = '%1.3f')

	plt.scatter(data1[:,0],data1[:,1], c='r', label='Class1')
	plt.scatter(data2[:,0],data2[:,1], c='b', label='Class2')
	plt.legend(loc='Upper right')
	plt.title('K=8 and Non Diagonal')
	plt.show()

	#Contour Plots for k templates Diagonal
	rv1 = []
	rv2 = []
	for i in range(no_templates):
		rv1 += [multivariate_normal(mu[0][i],cov_diag[0][i])]
		rv2 += [multivariate_normal(mu[1][i],cov_diag[1][i])]

	for i in range(no_templates):
		c1 = plt.contour(X,Y,rv1[i].pdf(val), colors='black')
		plt.clabel(c1, fmt = '%1.3f')

		c2 = plt.contour(X,Y,rv2[i].pdf(val), colors='magenta')
		plt.clabel(c2, fmt = '%1.3f')

	plt.scatter(data1[:,0],data1[:,1], c='r', label='Class1')
	plt.scatter(data2[:,0],data2[:,1], c='b', label='Class2')
	plt.legend(loc='Upper right')
	plt.title('K=8 and Diagonal')
	plt.show()
	

def analyzeImages():
	path = '/Users/pavan.ravishankar/Downloads/pr3ass/image_data/features/'
	types = ['coast/', 'mountain/', 'tallbuilding/']

	#[coast, mountain, tallbuilding]
	totaldata = []
	
	for tp in types:
		opath = path + tp
		files = [join(opath, f) for f in listdir(opath) if isfile(join(opath, f))]

		data = getData(files[0])
		files = files[1:]
		
		for fname in files:
			temp = getData(fname)
			data = np.concatenate((data, temp), axis=0)
		
		totaldata += [normalize(data)]	

	#Distoriton Method for approximate number of mixtures initialization
	
	dist1 = []
	dist2 = []
	dist3 = []
	
	max_clus = 20

	for k in range(max_clus):	
		kmeans1 = KMeans(n_clusters=k+1, random_state=0).fit(totaldata[0])
		kmeans2 = KMeans(n_clusters=k+1, random_state=0).fit(totaldata[1])
		kmeans3 = KMeans(n_clusters=k+1, random_state=0).fit(totaldata[2])

		dist1 += [kmeans1.inertia_/len(totaldata[0])]
		dist2 += [kmeans2.inertia_/len(totaldata[1])]
		dist3 += [kmeans3.inertia_/len(totaldata[2])]

	plt.plot(np.arange(max_clus)+1, dist1, label='Coast Data')
	plt.plot(np.arange(max_clus)+1, dist2, label='Mountain Data')
	plt.plot(np.arange(max_clus)+1, dist3, label='TallBuilding Data')
	plt.xlabel('Clusters')
	plt.ylabel('Distortion')
	plt.xticks(np.arange(max_clus)+1)
	plt.show()
	
	no_templates = 10
	em_iter = 5

	rows_per_image = 36

	classes = len(totaldata)

	train_data = []
	val_data = []
	for i in range(classes):
		tot_im = totaldata[i].shape[0] / rows_per_image
		train_data_size = (int) (0.7 * tot_im) * rows_per_image
		val_data_size = (int) (0.15 * tot_im) * rows_per_image
		train_data += [totaldata[i][0:train_data_size,:]]
		val_data += [totaldata[i][train_data_size:train_data_size+val_data_size,:]]

		
	train_acc = []
	val_acc = []
	train_accDiag = []
	val_accDiag = []
	
	train_accuracy, val_accuracy, train_accuracyDiag, val_accuracyDiag, _, _, _, _, _, _, _, _, _, _, trueResultVal, normtempDistVal, normtempDistValDiag, predResultValImg, predResultValDiagImg	 = buildGMM(train_data, val_data, no_templates, em_iter)
	plotConfROCDETRealData(trueResultVal, normtempDistVal, normtempDistValDiag, predResultValImg, predResultValDiagImg,10)
	
	for no_mix in range(1,17):
		train_accuracy, val_accuracy, train_accuracyDiag, val_accuracyDiag, _, _, _, _, _, _, _, _, _, _, trueResultVal, normtempDistVal, normtempDistValDiag, predResultValImg, predResultValDiagImg	 = buildGMM(train_data, val_data, no_mix, em_iter)
		train_acc += [train_accuracy]
		val_acc += [val_accuracy]

		train_accDiag += [train_accuracyDiag]
		val_accDiag += [val_accuracyDiag]

	axes = plt.gca()
	axes.set_axisbelow(True)
	axes.minorticks_on()
	axes.grid(which='major', linestyle='-', linewidth='0.5', color='red')	
	axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
	
	plt.scatter(np.arange(16)+1,val_accDiag, c='b', marker='+')
	plt.plot(np.arange(16)+1,val_accDiag, label='Real Data Validation Accuracy Diagonal')
	plt.scatter(np.arange(16)+1,train_accDiag, c='r', marker='+')
	plt.plot(np.arange(16)+1,train_accDiag, label='Real Data Training Accuracy Diagonal')
	plt.xlabel('No of Mixtures')
	plt.ylabel('Accuracy')
	plt.title('K max = 18 and EM steps = 5')
	plt.legend(loc='lower right')
	plt.show()
	
if __name__ == '__main__':
	analyzeSynthesizedData()
	analyzeImages()
