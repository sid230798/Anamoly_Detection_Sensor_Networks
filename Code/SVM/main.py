import numpy as np
from util import *


file_path = "./dataset/singlehop-indoor-moteid1-data.txt"

file_path1 = "./dataset/singlehop-outdoor-moteid4-data.txt"

indoor = read_file(file_path)

outdoor = read_file(file_path1)

#visualize_graph(indoor)
#visualize_graph(outdoor)



X_indoor,Y_indoor = get_XY(indoor)
X_outdoor,Y_outdoor = get_XY(outdoor)

indoor_ratio = int((len(Y_indoor)-np.sum(Y_indoor))/np.sum(Y_indoor))
outdoor_ratio = int((len(Y_outdoor)-np.sum(Y_outdoor))/np.sum(Y_outdoor))

#print(len(Y_indoor),np.sum(Y_indoor),len(Y_indoor)-np.sum(Y_indoor))
#print(len(Y_outdoor),np.sum(Y_outdoor),len(Y_outdoor)-np.sum(Y_outdoor))

clf = SVC(kernel='poly',class_weight={1: indoor_ratio}) 

#X_indoor = get_standardized_X(X_indoor)
#X_outdoor = get_standardized_X(X_outdoor)

#X_indoor = get_normalized_X(X_indoor)
#X_outdoor = get_normalized_X(X_outdoor)

#grad_X,grad_Y = get_gradient_form_data(X,Y)
#grad_X1,grad_Y1 = get_gradient_form_data(X1,Y1)

##visualize_with_readings(grad_X[:,0],grad_X[:,1],grad_Y)

#grad_X = get_standardized_X(grad_X)
#grad_X1 = get_standardized_X(grad_X1)

experiment3(X_indoor,Y_indoor,X_outdoor,Y_outdoor)