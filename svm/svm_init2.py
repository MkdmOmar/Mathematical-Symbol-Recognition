from sklearn import svm
import numpy as np
import scipy as sp
import sys
import time

testSamples = 'test.csv'
testLabels = 'testLabel.csv'
trainSamples = 'train.csv'
trainLabels = 'trainLabel.csv'

test_samples = []
train_samples = []

#sample_set = np.genfromtxt(samples,delimiter=1156)
test_labels = np.genfromtxt(testLabels,delimiter='\n',dtype=int)
train_labels = np.genfromtxt(trainLabels,delimiter='\n',dtype=int)

with open(testSamples) as f:
    for line in f:
    	test_samples.append(map(int,line.split()))

with open(trainSamples) as mu:
    for line in mu:
    	train_samples.append(map(int,line.split()))

#train multi-class SVM via one-vs-one
#better for unbalanced data / data with a ton of classes
""" Default settings for fit
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
"""
print "beginning ovo training"
start = time.time()
ovoSVM = svm.SVC(kernel='poly')
ovoSVM.fit(train_samples, train_labels)
end = time.time()
print "time to train: ", (end-start)

print "get support_vectors_"
print ovoSVM.support_vectors_
print "number of support vectors per class"
print ovoSVM.n_support_

#(n_class * n_class - 1) / 2 (by) n_features
#coefficients for each classifier (how much each feature is important)
# each classifier (row) for a pair of classes 
""" Only available for linear kernel
print "get coefficients"
print ovoSVM.coef_
print "get intercepts"
print ovoSVM.intercept_
print "get dual coefficients"
print ovoSVM.dual_coef_
"""

predictions = ovoSVM.predict(test_samples)
predictions = np.array(predictions)
test_labels = np.array(test_labels)
print "percent error rate: ", np.mean(predictions != test_labels)

#train multi-class SVM via one-vs-rest
#better for small number of classes like digits (we have 24 classes)
""" Default Constructor
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
"""

print "beginning ovr training"
start = time.time()
ovrSVM = svm.LinearSVC(C=0.5)
ovrSVM.fit(train_samples, train_labels)
end = time.time()
print "time to train: ", (end-start)

"""not available for LinearSVC
print "get support_vectors_"
print ovrSVM.support_vectors_
print "number of support vectors per class"
print ovrSVM.n_support_
"""

# n_class (by) n_features
# coefficients for each classifier (how much each feature is important)
# each classifier (row) for a single class
print "get coefficients, (0,1) (0,2) ... (1,2) (1,3) ..."
print ovrSVM.coef_
print "get intercepts"
print ovrSVM.intercept_

#santiy_check
#print test_samples[1]
#print type(test_samples[1][1])

predictions = ovrSVM.predict(test_samples)
predictions = np.array(predictions)
test_labels = np.array(test_labels)

if len(predictions) != len(test_labels):
	print 'Length Error'
print "percent error rate: ", np.mean(predictions != test_labels)
for i in range(0,4):
	print test_labels[i], predictions[i]


#TODO
#five-fold cross validation
"""
fifth = int(0.2 * length)

for i in range(0,5):

	train_start = (fifth * i)
	if (i == 4):
		train_stop = length
	else:
		train_stop =  (fifth * (i+1))

	train_labels = [label_set[i] for i in indexes[train_start:train_stop]]
	test_labels = [label_set[i] for i in indexes[0:train_start]]
	for i in range(train_stop, length): 
		test_labels.append(indexes[i])

	train_samples = [label_set[i] for i in indexes[train_start:train_stop]]
	test_samples = [label_set[i] for i in indexes[0:train_start]]
	for i in range(train_stop, length):
		test_samples.append(indexes[i])
"""
#TODO
#train ovoSVM with different kernels

#save to disk 
#to retrieve-> ovoSVM = joblib.load("ovoSVM.pkl")
#from sklearn.externals import joblib
#joblib.dump(ovoSVM,"ovoSVM.pkl")
#joblib.dump(ovrSVM,"ovrSVM.pkl")



