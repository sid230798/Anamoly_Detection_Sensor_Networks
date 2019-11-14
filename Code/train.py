import numpy as np
from util import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 

train_file = "./../dataset/singlehop-indoor-moteid2-data.txt"
test_file = "./../dataset/singlehop-indoor-moteid1-data.txt"

trainSet = read_file(train_file)
testSet = read_file(test_file)

trainX, trainY = get_XY(trainSet)
testX, testY = get_XY(testSet)

trainX1, trainX2 = (trainX[:, 1]).reshape(-1, 1), (trainX[:, 0]).reshape(-1, 1)
testX1, testX2 = (testX[:, 1]).reshape(-1, 1), (testX[:, 0]).reshape(-1, 1)
#print(trainX1.shape, trainX2.shape)
#print(np.where(testY == 1))

scaler = StandardScaler()
testX1, testX2 = scaler.fit_transform(testX1), scaler.fit_transform(testX2)
visualize(testX1, testY)