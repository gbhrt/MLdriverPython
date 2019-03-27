import numpy as np
import library as lib
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing

def train_model():
    
    plt.ion()
    losses = []
    for i in range(100000):
        if waitFor.stop == [True]:
            break
        net.update_network(Xtrain,ytrain_,keep_prob = keep_prob)
        if i % 100 == 0:
            loss = net.get_loss(Xtrain,ytrain_)
            print(loss)
            losses.append(loss)

            plt.cla()
            plt.plot(losses)
            plt.plot(lib.running_average(losses,50))
            plt.draw()
            plt.pause(0.0001)
    plt.ioff()        
    plt.show()
    net.save_model(os.getcwd()+"\\files\\",name = save_name)

if __name__ == "__main__": 
    waitFor = lib.waitFor()

    save_name = "uncertainty1"
    restore_flag = True
    train_flag = False
    X_n = 1
    Y_n = 1
    alpha = 0.0001
    keep_prob = 0.8


    n = 50
    mu, sigma1,sigma2 = 0, 0.01, 0.2 # mean and standard deviation
    #mu, sigma1,sigma2 = 0, 0.0, 0.0 # mean and standard deviation
    noise = np.append(np.random.normal(mu, sigma1, int(n/2)),np.random.normal(mu, sigma2, int(n/2)))
    x =  np.linspace(0, 10, n)
    Xtrain = x
    ytrain_ = np.sin(Xtrain)+noise

    for i in range(10):
        noise = np.append(np.random.normal(mu, sigma1, int(n/2)),np.random.normal(mu, sigma2, int(n/2)))
        Xtrain = np.append(Xtrain,x)
        ytrain_ = np.append(ytrain_, np.sin(x)+noise)

    plt.figure(1)
    plt.plot(Xtrain, ytrain_,"o",label = "y_ train")
    plt.show()  

    Xtrain = Xtrain.reshape(-1,1)
    ytrain_ = ytrain_.reshape(-1,1)

    Xtest = np.linspace(0, 40, n).reshape(-1,1)
    ytest_ = np.sin(Xtest)

    #plt.plot(Xtrain, ytrain_,"o",label = "y_ train")
    #plt.show()
    scalerX = preprocessing.StandardScaler().fit(Xtest)
    scalerY = preprocessing.StandardScaler().fit(ytrain_)
    Xtrain = scalerX.transform(Xtrain)
    ytrain_ = scalerY.transform(ytrain_)
    Xtest = scalerX.transform(Xtest)
    ytest_ = scalerY.transform(ytest_)
    from model_based_net import model_based_network

    net = model_based_network(X_n,Y_n,alpha,net_type = 2)#,dropout_flag = True

    if restore_flag:
        net.restore(os.getcwd()+"\\files\\",name = save_name)
    if train_flag:
        train_model()
    ytest = []

    
    T = 10

    ytrain = (net.get_Y(Xtrain))

    plt.plot(Xtrain, ytrain,"o",label = "y train")
    plt.plot(Xtrain, ytrain_,"o",label = "y_ train")
    plt.plot(Xtest, ytest_, label = "y_ test")
    for i in range(T):
        ytest = (net.get_Y(Xtest,keep_prob = keep_prob))
        plt.plot(Xtest, ytest, label = "y test",alpha = 0.7)
    
    plt.legend()
    plt.show()
