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
        net.update_network(Xtrain,ytrain_)
        if i % 100 == 0:
            loss = net.get_loss(Xtrain,ytrain_)
            print(loss)
            losses.append(loss)

            plt.cla()
            plt.plot(losses)
           # plt.plot(lib.running_average(losses,50))
            plt.draw()
            plt.pause(0.0001)
    plt.ioff()        
    plt.show()

    #file_name =  path+name+".ckpt "
    net.model.save_weights(os.getcwd()+"\\files\\"+save_name)
    #net.save_model(os.getcwd()+"\\files\\",name = save_name)

if __name__ == "__main__": 
    waitFor = lib.waitFor()

    save_name = "rect_func.ckpt"
    restore_flag = False
    train_flag = True
    X_n = 1
    Y_n = 1
    alpha = 0.0001


    n = 50

    x =  np.linspace(0, 10, n)
    Xtrain = x
    ytrain_ = []
    for i in range(len(x)):
        ytrain_.append(0 if i<10 or i>30 else 1)
    ytrain_ = np.array(ytrain_)

    #plt.figure(1)
    #plt.plot(Xtrain, ytrain_,"o",label = "y_ train")
    #plt.show()  

    Xtrain = Xtrain.reshape(-1,1)
    ytrain_ = ytrain_.reshape(-1,1)

    #Xtest = np.linspace(0, 40, n).reshape(-1,1)
    #ytest_ = np.sin(Xtest)

    #plt.plot(Xtrain, ytrain_,"o",label = "y_ train")
    #plt.show()
    scalerX = preprocessing.StandardScaler().fit(Xtrain)
    scalerY = preprocessing.StandardScaler().fit(ytrain_)
    Xtrain = scalerX.transform(Xtrain)
    ytrain_ = scalerY.transform(ytrain_)
    #Xtest = scalerX.transform(Xtest)
    #ytest_ = scalerY.transform(ytest_)
    from model_based_net import model_based_network

    net = model_based_network(X_n,Y_n,alpha)#,dropout_flag = True

    if restore_flag:
        net.restore(os.getcwd()+"\\files\\",name = save_name)
    if train_flag:
        train_model()
   
    #ytest = []


    ytrain = (net.get_Y(Xtrain))

    plt.plot(Xtrain, ytrain,"o",label = "y train")
    plt.plot(Xtrain, ytrain_,"o",label = "y_ train")
    #plt.plot(Xtest, ytest_, label = "y_ test")

    
    plt.legend()
    plt.show()
