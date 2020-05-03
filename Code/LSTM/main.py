import numpy as np
from util import *
from sklearn.svm import SVC # "Support Vector Classifier" 


file_path = "./../dataset/singlehop-indoor-moteid1-data.txt"
file_path1 = "./../dataset/singlehop-outdoor-moteid4-data.txt"

test = read_file(file_path)
test1 = read_file(file_path1)

#visualize_graph(test)
visualize_graph(test1)

clf = SVC(kernel='rbf') 

X,Y = get_XY(test)
X1,Y1 = get_XY(test1)

X = get_standardized_X(X)
X1 = get_standardized_X(X1)

X = get_normalized_X(X)
X1 = get_normalized_X(X1)

grad_X,grad_Y = get_gradient_form_data(X,Y)
grad_X1,grad_Y1 = get_gradient_form_data(X1,Y1)

#visualize_with_readings(grad_X[:,0],grad_X[:,1],grad_Y)

grad_X = get_standardized_X(grad_X)
grad_X1 = get_standardized_X(grad_X1)

print("Training")
clf.fit(X, Y) 
print("Trained")

print(X.shape,Y.shape)
print(X1.shape,Y1.shape)
err,per_err = get_error(clf,X,Y)
print("Indoor : Total err was ", err, " And percentage ",per_err)
err,per_err = get_error(clf,X1,Y1)
print("Outdoor: Total err was ", err, " And percentage ",per_err)
