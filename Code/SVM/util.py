import numpy as np
import matplotlib.pyplot as plt 
from sklearn.svm import SVC # "Support Vector Classifier" 

def read_file(file_path):
    with open(file_path) as f:
        content = f.readlines()
    content = [x.strip() for x in content] 
    content = content[4:]
    output = []
    for line in content:
        line = line.split("\t")
        reading_id     =     int(line[0])
        mote_id     =     int(line[1])
        humidity     =     float(line[2])
        temp         =     float(line[3])
        label         =     int(line[4])
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

def visualize(readings,prop, label,prop_text):
    plt.clf()
    plt.plot(readings,prop)
    false_data = readings*label
    false_data = np.where(false_data!=0)[0]
    y_upper = np.max(false_data)
    y_lower = np.min(false_data)
    x_lower = int(np.min(prop))
    x_upper = int(np.max(prop))
    coloring = list(range(x_lower,x_upper+1))
    plt.fill_betweenx(coloring,y_upper,y_lower,alpha = 0.2,color = 'r')
    plt.xlabel("Time",fontsize = 20)
    plt.ylabel(prop_text, fontsize = 20)
    plt.legend()
    plt.show()

def get_positive_negative_data(data,label):
    positive_data_index = data*(1-label)
    positive_data_index = np.where(positive_data_index!=0)[0]

    negative_data_index = data*(label)
    negative_data_index = np.where(negative_data_index!=0)[0]

    positive_data = list()
    negative_data = list()
    for index in positive_data_index:
        positive_data.append(data[index])
    for index in negative_data_index:
        negative_data.append(data[index])
    return np.asarray(positive_data),np.asarray(negative_data)


def scatter(prop1,prop2,labels,prop1_text,prop2_text):
    plt.clf()
    positive_prop1,negative_prop1 = get_positive_negative_data(prop1,labels)
    positive_prop2,negative_prop2 = get_positive_negative_data(prop1,labels)
    plt.scatter(positive_prop1,positive_prop2,color = 'g',label = "Valid Data")
    plt.scatter(negative_prop1,negative_prop2,color = 'r',label = "Negative Data")

    plt.xlabel(prop1_text, fontsize = 20)
    plt.ylabel(prop2_text, fontsize = 20)
    plt.legend(prop={'size': 18})
    plt.show()

def visualize_with_readings(temp,humidity,label,readings = None):
    try:
        if readings==None:
            readings = list(range(1,len(humidity)+1))
    except:
        pass
    print(len(readings),len(temp),len(humidity),len(label))
    visualize(readings,humidity,label,"Humidity")
    visualize(readings,temp,label,"Temperature")
    scatter(humidity,temp,label,"Humidity","Temperature")


def visualize_graph(input_data):
    plt.clf()
    readings =  input_data[:,[0]].T[0]
    temp     =  input_data[:,[3]].T[0]
    humidity =  input_data[:,[2]].T[0]
    label    =  input_data[:,[4]].T[0]
    #visualize(temp.T,label)
    visualize_with_readings(temp,humidity,label,readings)

def get_error(model,X,Y):
    predicted_Y = model.predict(X)
    error = Y-predicted_Y
    erro = error.tolist()
    total_err = np.sum(np.abs((error)))
    percentage = float(total_err)/len(Y)
    return total_err,percentage

def experiment1(X_indoor,Y_indoor,X_outdoor,Y_outdoor):
    indoor_ratio = int((len(Y_indoor)-np.sum(Y_indoor))/np.sum(Y_indoor))
    outdoor_ratio = int((len(Y_outdoor)-np.sum(Y_outdoor))/np.sum(Y_outdoor))

    indoor_poly_clf = SVC(kernel='poly',class_weight={1: indoor_ratio}) 
    indoor_linear_clf = SVC(kernel='linear',class_weight={1: indoor_ratio}) 
    indoor_rbf_clf = SVC(kernel='rbf',class_weight={1: indoor_ratio}) 
    indoor_sigmoid_clf = SVC(kernel='rbf',class_weight={1: indoor_ratio}) 

    outdoor_poly_clf = SVC(kernel='poly',class_weight={1: outdoor_ratio}) 
    outdoor_linear_clf = SVC(kernel='linear',class_weight={1: outdoor_ratio}) 
    outdoor_rbf_clf = SVC(kernel='rbf',class_weight={1: outdoor_ratio}) 
    outdoor_sigmoid_clf = SVC(kernel='rbf',class_weight={1: outdoor_ratio}) 

    "Training"
    indoor_poly_clf.fit(X_indoor,Y_indoor)
    indoor_linear_clf.fit(X_indoor,Y_indoor)
    indoor_rbf_clf.fit(X_indoor,Y_indoor)
    indoor_sigmoid_clf.fit(X_indoor,Y_indoor)

    outdoor_poly_clf.fit(X_outdoor,Y_outdoor)
    outdoor_linear_clf.fit(X_outdoor,Y_outdoor)
    outdoor_rbf_clf.fit(X_outdoor,Y_outdoor)
    outdoor_sigmoid_clf.fit(X_outdoor,Y_outdoor)
    "Trained Successfully"

    #Indoor Errors
    _,per_err1 = get_error(indoor_poly_clf,X_indoor,Y_indoor)
    _,per_err2 = get_error(indoor_linear_clf,X_indoor,Y_indoor)
    _,per_err3 = get_error(indoor_rbf_clf,X_indoor,Y_indoor)
    _,per_err4 = get_error(indoor_sigmoid_clf,X_indoor,Y_indoor)

    _,per_err5 = get_error(outdoor_poly_clf,X_indoor,Y_indoor)
    _,per_err6 = get_error(outdoor_linear_clf,X_indoor,Y_indoor)
    _,per_err7 = get_error(outdoor_rbf_clf,X_indoor,Y_indoor)
    _,per_err8 = get_error(outdoor_sigmoid_clf,X_indoor,Y_indoor)

    #Outdoor Errors

    _,per_err9 = get_error(indoor_poly_clf,X_outdoor,Y_outdoor)
    _,per_err10 = get_error(indoor_linear_clf,X_outdoor,Y_outdoor)
    _,per_err11 = get_error(indoor_rbf_clf,X_outdoor,Y_outdoor)
    _,per_err12 = get_error(indoor_sigmoid_clf,X_outdoor,Y_outdoor)

    _,per_err13 = get_error(outdoor_poly_clf,X_outdoor,Y_outdoor)
    _,per_err14 = get_error(outdoor_linear_clf,X_outdoor,Y_outdoor)
    _,per_err15 = get_error(outdoor_rbf_clf,X_outdoor,Y_outdoor)
    _,per_err16 = get_error(outdoor_sigmoid_clf,X_outdoor,Y_outdoor)

    print("Trained on Indoor\t on Indoor Dataset"," with kernel : linear ->\t\t\t\t",per_err2*100)
    print("Trained on Indoor\t on Indoor Dataset"," with kernel : Polynomial ->\t\t\t\t",per_err1*100)
    print("Trained on Indoor\t on Indoor Dataset"," with kernel : Sigmoid ->\t\t\t\t",per_err4*100)
    print("Trained on Indoor\t on Indoor Dataset"," with kernel : Radial basis Function ->\t\t",per_err3*100)

    print("Trained on Outdoor\t on Indoor Dataset"," with kernel : linear ->\t\t\t\t",per_err6*100)
    print("Trained on Outdoor\t on Indoor Dataset"," with kernel : Polynomial ->\t\t\t\t",per_err5*100)
    print("Trained on Outdoor\t on Indoor Dataset"," with kernel : Sigmoid ->\t\t\t\t",per_err8*100)
    print("Trained on Outdoor\t on Indoor Dataset"," with kernel : Radial basis Function ->\t\t",per_err7*100)

    print("Trained on Indoor\t on Outdoor Dataset"," with kernel : linear ->\t\t\t\t",per_err10*100)
    print("Trained on Indoor\t on Outdoor Dataset"," with kernel : Polynomial ->\t\t\t",per_err9*100)
    print("Trained on Indoor\t on Outdoor Dataset"," with kernel : Sigmoid ->\t\t\t\t",per_err12*100)
    print("Trained on Indoor\t on Outdoor Dataset"," with kernel : Radial basis Function ->\t\t",per_err11*100)

    print("Trained on Outdoor\t on Outdoor Dataset"," with kernel : linear ->\t\t\t\t",per_err14*100)
    print("Trained on Outdoor\t on Outdoor Dataset"," with kernel : Polynomial ->\t\t\t",per_err13*100)
    print("Trained on Outdoor\t on Outdoor Dataset"," with kernel : Sigmoid ->\t\t\t\t",per_err16*100)
    print("Trained on Outdoor\t on Outdoor Dataset"," with kernel : Radial basis Function ->\t\t",per_err15*100)

def experiment2(X_outdoor,Y_outdoor):
    shifts = list(range(-100,100,10))

    outdoor_ratio = int((len(Y_outdoor)-np.sum(Y_outdoor))/np.sum(Y_outdoor))
    outdoor_poly_clf = SVC(kernel='poly',class_weight={1: outdoor_ratio}) 
    outdoor_linear_clf = SVC(kernel='linear',class_weight={1: outdoor_ratio}) 
    outdoor_rbf_clf = SVC(kernel='rbf',class_weight={1: outdoor_ratio}) 
    outdoor_sigmoid_clf = SVC(kernel='rbf',class_weight={1: outdoor_ratio}) 

    outdoor_poly_clf.fit(X_outdoor,Y_outdoor)
    outdoor_linear_clf.fit(X_outdoor,Y_outdoor)
    outdoor_rbf_clf.fit(X_outdoor,Y_outdoor)
    outdoor_sigmoid_clf.fit(X_outdoor,Y_outdoor)

    linear_errs = []
    polynomial_errs = []
    rbf_errs = []
    sigmoid_errs = []

    for shift in shifts:
        _,err_per = get_error(outdoor_linear_clf,X_outdoor-shift,Y_outdoor)
        linear_errs.append(err_per*len(Y_outdoor))
    for shift in shifts:
        _,err_per = get_error(outdoor_poly_clf,X_outdoor-shift,Y_outdoor)
        polynomial_errs.append(err_per*len(Y_outdoor))
    for shift in shifts:
        _,err_per = get_error(outdoor_rbf_clf,X_outdoor-shift,Y_outdoor)
        rbf_errs.append(err_per*len(Y_outdoor))
    for shift in shifts:
        _,err_per = get_error(outdoor_sigmoid_clf,X_outdoor-shift,Y_outdoor)
        sigmoid_errs.append(err_per*len(Y_outdoor))
    
    plt.clf()
    plt.plot(shifts,linear_errs,label = "Linear")
    plt.plot(shifts,polynomial_errs,label = "Polynomial")
    plt.plot(shifts,rbf_errs,label = "RBF")
    plt.plot(shifts,sigmoid_errs,label = "Sigmoid")
    plt.legend()
    plt.xlabel("shift in data")
    plt.ylabel("Number of misclassified data")
    plt.show()

def experiment3(X_indoor,Y_indoor,X_outdoor,Y_outdoor):
    shifts = list(range(-100,100,10))

    grad_X,grad_Y = get_gradient_form_data(X_outdoor,Y_outdoor)
    outdoor_ratio = int((len(grad_Y)-np.sum(grad_Y))/np.sum(grad_Y))

    outdoor_poly_clf = SVC(kernel='poly',class_weight={1: outdoor_ratio}) 
    outdoor_linear_clf = SVC(kernel='linear',class_weight={1: outdoor_ratio}) 
    outdoor_rbf_clf = SVC(kernel='rbf',class_weight={1: outdoor_ratio}) 
    outdoor_sigmoid_clf = SVC(kernel='rbf',class_weight={1: outdoor_ratio})

    outdoor_poly_clf.fit(grad_X,grad_Y)
    outdoor_linear_clf.fit(grad_X,grad_Y)
    outdoor_rbf_clf.fit(grad_X,grad_Y)
    outdoor_sigmoid_clf.fit(grad_X,grad_Y)

    linear_errs = []
    polynomial_errs = []
    rbf_errs = []
    sigmoid_errs = []

    for shift in shifts:
        _,err_per = get_error(outdoor_linear_clf,grad_X-shift,grad_Y)
        linear_errs.append(err_per*len(Y_outdoor))
    for shift in shifts:
        _,err_per = get_error(outdoor_poly_clf,grad_X-shift,grad_Y)
        polynomial_errs.append(err_per*len(Y_outdoor))
    for shift in shifts:
        _,err_per = get_error(outdoor_rbf_clf,grad_X-shift,grad_Y)
        rbf_errs.append(err_per*len(Y_outdoor))
    for shift in shifts:
        _,err_per = get_error(outdoor_sigmoid_clf,grad_X-shift,grad_Y)
        sigmoid_errs.append(err_per*len(Y_outdoor))

    plt.clf()
    plt.plot(shifts,linear_errs,label = "Linear")
    plt.plot(shifts,polynomial_errs,label = "Polynomial")
    plt.plot(shifts,rbf_errs,label = "RBF")
    plt.plot(shifts,sigmoid_errs,label = "Sigmoid")
    plt.legend()
    plt.title("Outdoor Data")
    plt.xlabel("shift in data")
    plt.ylabel("Number of misclassified data")
    plt.show()

    linear_errs = []
    polynomial_errs = []
    rbf_errs = []
    sigmoid_errs = []

    grad_X,grad_Y = get_gradient_form_data(X_indoor,Y_indoor)

    for shift in shifts:
        _,err_per = get_error(outdoor_linear_clf,grad_X-shift,grad_Y)
        linear_errs.append(err_per*len(Y_outdoor))
    for shift in shifts:
        _,err_per = get_error(outdoor_poly_clf,grad_X-shift,grad_Y)
        polynomial_errs.append(err_per*len(Y_outdoor))
    for shift in shifts:
        _,err_per = get_error(outdoor_rbf_clf,grad_X-shift,grad_Y)
        rbf_errs.append(err_per*len(Y_outdoor))
    for shift in shifts:
        _,err_per = get_error(outdoor_sigmoid_clf,grad_X-shift,grad_Y)
        sigmoid_errs.append(err_per*len(Y_outdoor))

    plt.clf()
    plt.plot(shifts,linear_errs,label = "Linear")
    plt.plot(shifts,polynomial_errs,label = "Polynomial")
    plt.plot(shifts,rbf_errs,label = "RBF")
    plt.plot(shifts,sigmoid_errs,label = "Sigmoid")
    plt.legend()
    plt.xlabel("shift in data")
    plt.title("Indoor Data")
    plt.ylabel("Number of misclassified data")
    plt.show()


