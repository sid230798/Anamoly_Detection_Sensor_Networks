import numpy as np
import matplotlib.pyplot as plt 


def read_file(file_path):
	with open(file_path) as f:
		content = f.readlines()
	content = [x.strip() for x in content] 
	content = content[4:]
	output = []
	for line in content:
		line = line.split("\t")
		reading_id 	= 	int(line[0])
		mote_id 	= 	int(line[1])
		humidity 	= 	float(line[2])
		temp 		= 	float(line[3])
		label 		= 	int(line[4])
		output.append([reading_id,mote_id,humidity,temp,label])
	output = np.asarray(output)
	return output
	
def get_XY(matrix):
	X = matrix[:,[2,3]]
	Y = matrix[:,[4]].T[0]
	Y = Y.astype(int)
	return X,Y

def get_normalized_X(X):
	return (X - np.amin(X, axis=0))/np.amax(X, axis=0)

def get_standardized_X(X):
	return (X - np.mean(X, axis=0))/np.std(X, axis=0)


def get_gradient_form_data(X,Y):
	X_tmp = X[1:] - X[:-1] 
	return X_tmp,Y[1:]
	
def visualize_with_readings(temp,humidity,label,readings = None):
	try:
		if readings==None:
			readings = list(range(1,len(humidity)+1))
	except:
		pass
	print(len(readings),len(temp),len(humidity),len(label))
	positive_humidity = humidity*(1-label)
	positive_readings = readings*(1-label)
	negative_humidity = humidity*label
	positive_temp = temp*(1-label)
	negative_temp = temp*label
	plt.plot(readings, positive_humidity, label = "Valid data") 
	plt.plot(readings, negative_humidity, label = "Invalid data") 
	plt.legend() 
	plt.show()
	plt.clf()
	plt.plot(readings, positive_temp, label = "Valid data") 
	plt.plot(readings, negative_temp, label = "Invalid data") 
	plt.legend() 
	plt.show()


def visualize_graph(input_data):
	plt.clf()
	readings =  input_data[:,[0]].T[0]
	temp     =  input_data[:,[3]].T[0]
	humidity =  input_data[:,[2]].T[0]
	label    =  input_data[:,[4]].T[0]
	visualize_with_readings(temp,humidity,label,readings)

def get_error(model,X,Y):
	predicted_Y = model.predict(X)
	error = Y-predicted_Y
	erro = error.tolist()
	total_err = np.sum(np.abs((error)))
	percentage = float(total_err)/len(Y)
	return total_err,percentage
