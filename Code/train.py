import numpy as np
from util import *
import matplotlib.pyplot as plt
import torch
from model import model
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split

train_file = "./../dataset/singlehop-outdoor-moteid3-data.txt"
test_file = "./../dataset/singlehop-indoor-moteid2-data.txt"

trainSet = read_file(train_file)
testSet = read_file(test_file)

trainX, trainY = get_XY(trainSet)
testX, testY = get_XY(testSet)

#trainX1, trainX2 = (trainX[:, 1]).reshape(-1, 1), (trainX[:, 0]).reshape(-1, 1)
#print(trainX.shape)
#print(np.where(testY == 1))

scaler = StandardScaler()
#trainX1, trainX2 = scaler.fit_transform(trainX1), scaler.fit_transform(trainX2)
#testX1, testX2 = scaler.fit_transform(testX1), scaler.fit_transform(testX2)

trainX, testX = scaler.fit_transform(trainX), scaler.fit_transform(testX)
testX1, testX2 = (testX[:, 1]).reshape(-1, 1), (testX[:, 0]).reshape(-1, 1)

input_features = 2
seq_len = 10
pred_len = 3
hidden_units = 35
batch_size = 32

framework = model(input_features, hidden_units, seq_len, pred_len)
framework.init_hidden(batch_size)
optimizer = torch.optim.Adam([{'params' : framework.parameters()}], lr=0.0001)
loss_fn = torch.nn.MSELoss()

trainSeqX, trainLabelX = create_subseq(trainX, seq_len, pred_len)
testSeqX, testLabelX = create_subseq(testX, seq_len, pred_len)
X_train, X_test, y_train, y_test = train_test_split(trainSeqX, trainLabelX, test_size=0.2)
X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)

train_size = X_train.shape[0]
test_size = X_test.shape[0]
#print(X_train.shape, y_train.shape)
#visualize(trainX1, trainY)
period = 10
max_epoch = 1000
loss_prev = np.inf
best_loss = np.inf
train_loss_list = []
test_loss_list = []
isTrain = False

if isTrain == True :
    for epoch in range(max_epoch):

        perm = np.random.permutation(train_size)
        train_loss = 0

        for i in range(train_size//batch_size) :

            optimizer.zero_grad()
            framework.init_hidden(batch_size)
            batch_x = X_train[perm[i*batch_size:(i+1)*batch_size]]
            batch_y = y_train[perm[i*batch_size:(i+1)*batch_size]]
            
            pred_y_temp, pred_y_humid = framework(batch_x)
            loss1 = loss_fn(pred_y_humid, batch_y[:, :, 0])
            loss2 = loss_fn(pred_y_temp, batch_y[:, :, 1])
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            train_loss += loss
        
        train_loss /= (train_size//batch_size)
        train_loss = train_loss.detach().numpy()
        train_loss_list.append(train_loss)


        perm_test = np.random.permutation(test_size)
        test_loss = 0
        for i in range(test_size//batch_size) :

            framework.init_hidden(batch_size)
            batch_x = X_test[perm_test[i*batch_size:(i+1)*batch_size]]
            batch_y = y_test[perm_test[i*batch_size:(i+1)*batch_size]]
            
            pred_y_temp, pred_y_humid = framework(batch_x)
            loss1 = loss_fn(pred_y_humid, batch_y[:, :, 0])
            loss2 = loss_fn(pred_y_temp, batch_y[:, :, 1])
            loss = loss1 + loss2

            test_loss += loss
        
        test_loss /= (test_size//batch_size)
        test_loss = test_loss.detach().numpy()
        test_loss_list.append(test_loss)

        # check early stopping
        if epoch % period == 0:
            print('epoch:{} train loss:{} test loss:{}'.format(epoch, train_loss, test_loss))
            if(best_loss > test_loss) :
                torch.save(framework.state_dict(), "./Model_Outdoor/model-ckpt-best.txt")
                best_loss = test_loss
            torch.save(framework.state_dict(), "./Model_Outdoor/model-ckpt.txt")
            loss_prev = test_loss
else :

    batch_size = 1
    framework.load_state_dict(torch.load("./Model_Outdoor/model-ckpt-best.txt"))
    framework.eval()
    framework.init_hidden(batch_size)

    for param in framework.parameters():
        param.requires_grad = False

    temp_error_list = []
    humid_error_list = []
    for i in range(test_size//batch_size) :
        
        framework.init_hidden(batch_size)
        batch_x = X_test[i*batch_size:(i+1)*batch_size]
        batch_y = y_test[i*batch_size:(i+1)*batch_size]
            
        pred_y_temp, pred_y_humid = framework(batch_x)
        temp_error_list.append(batch_y[:, :, 1] - pred_y_temp)
        humid_error_list.append(batch_y[:, :, 0] - pred_y_humid)
        #print(batch_x.shape, batch_y.shape, pred_y_humid.shape, pred_y_temp.shape)

    temp_error = torch.stack(temp_error_list)
    humid_error = torch.stack(humid_error_list)

    temp_mean_error = torch.mean(temp_error.squeeze(dim=1), dim=0)
    humid_mean_error = torch.mean(humid_error.squeeze(dim=1), dim=0)

    temp_cov = temp_error - temp_mean_error
    humid_conv = humid_error - humid_mean_error

    temp_cov = torch.mean(torch.matmul(temp_cov.view(test_size, -1, 1), temp_cov), dim=0)
    humid_cov = torch.mean(torch.matmul(humid_conv.view(test_size, -1, 1), humid_conv), dim=0)

    print("Mean => ",temp_mean_error, " \nCovariance Matrix => ",temp_cov)
    ## ------------------------------------------------------------------------------------------------
    # Calculate Mahabolis distance for unseen data
    ## ------------------------------------------------------------------------------------------------
    eval_size = testSeqX.shape[0]
    eval_Temp_error_list = []
    eval_Humid_error_list = []

    testSeqX, testLabelX = torch.Tensor(testSeqX), torch.Tensor(testLabelX)
    #print(testSeqX.shape, testLabelX.shape)
     
    for i in range(eval_size//batch_size) :
        
        framework.init_hidden(batch_size)
        batch_x = testSeqX[i*batch_size:(i+1)*batch_size]
        batch_y = testLabelX[i*batch_size:(i+1)*batch_size]
            
        pred_y_temp, pred_y_humid = framework(batch_x)
        eval_Temp_error_list.append(batch_y[:, :, 1] - pred_y_temp)
        eval_Humid_error_list.append(batch_y[:, :, 0] - pred_y_humid)

    temp_error = torch.stack(eval_Temp_error_list)
    humid_error = torch.stack(eval_Humid_error_list)
    #print(eval_size, temp_error.shape, humid_error.shape)
    
    mahbolis_distance_temp = torch.matmul(torch.matmul((temp_error - temp_mean_error), torch.inverse(temp_cov)), (temp_error - temp_mean_error).view(eval_size, -1, 1)).squeeze(dim=2).numpy()
    mahbolis_distance_humid = torch.matmul(torch.matmul((humid_error - humid_mean_error), torch.inverse(humid_cov)), (humid_error - humid_mean_error).view(eval_size, -1, 1)).squeeze(dim=2).numpy()  

    #print(testX1.shape)
    #print(mahbolis_distance_temp.shape, mahbolis_distance_humid.shape)
    #visualize_duel(testX1[seq_len+pred_len:], mahbolis_distance_temp, testY[seq_len+pred_len:])
    visualize_duel(testX2[seq_len+pred_len:], mahbolis_distance_humid, testY[seq_len+pred_len:])


