import test_net_performance
import library as lib
import hyper_parameters
import matplotlib.pyplot as plt
import numpy as np
import environment1 
import json
import os
import random
from sklearn import preprocessing
import agent_lib as pLib

if __name__ == "__main__": 
    restore = True
    train = True
    split_buffer = True
    separate_nets = True

    scaling_type ="scaler" # "scaler"#standard_scaler
    test_part = 0.3
    num_train = 10000000

    description = "small_state_var"
    file_name = "small_state_var.txt"


    waitFor = lib.waitFor()
    HP = hyper_parameters.ModelBasedHyperParameters()
    envData = environment1.OptimalVelocityPlannerData('model_based')

    if separate_nets:
        from model_based_net_sep import model_based_network
    else:
        from model_based_net import model_based_network
    #net = model_based_network(envData.X_n,envData.Y_n,HP.alpha)

    invNet= model_based_network(envData.X_n,1,HP.alpha)#input [vel_y,steer,roll,acc_action,roll_next], output [steer command]

    Replay = pLib.Replay(1000000)
    #velocity, steering angle, steering action, acceleration action,  rel x, rel y, rel ang, velocity next, steering next, roll
    Replay.restore(HP.restore_file_path)
    #Replay.memory = Replay.memory[:10000]
    print("length of buffer: ",len(Replay.memory))

    #state, a, next_state, end,_ = map(list, zip(*Replay.memory))#all data
    vec_X, vec_Y_, end,_ = map(list, zip(*Replay.memory))
    
    inv_vec_X,inv_vec_Y_ = [],[]
    for x,y in zip(vec_X, vec_Y_):
        inv_vec_X.append([x[0],x[1],x[2],x[4],x[2]+y[5]])#self.X_names = ["vel_y","steer","roll","steer_action","acc_action"]
        inv_vec_Y_.append([x[3]])
    inv_train_X,inv_train_Y_,inv_test_X,inv_test_Y_ = test_net_performance.scale_and_split_data(scaling_type,test_part,inv_vec_X,inv_vec_Y_)

    train_X,train_Y_,test_X,test_Y_ = test_net_performance.scale_and_split_data(scaling_type,test_part,vec_X,vec_Y_)

    if restore:
        #net.restore(HP.restore_file_path)
        invNet.restore(HP.restore_file_path)

    
    if train:
        #test_net_performance.train_net(HP,net,train_X,train_Y_,envData, waitFor,num_train,separate_nets)#ReplayTrain
        test_net_performance.train_net(HP,invNet,inv_train_X,inv_train_Y_,envData, waitFor,num_train,separate_nets)#ReplayTrain
    
    print("test loss: ",invNet.get_loss(inv_test_X,inv_test_Y_))
    print("train loss: ",invNet.get_loss(inv_train_X,inv_train_Y_))

    inv_train_Y = invNet.get_Y(inv_train_X).tolist()
    inv_test_Y = invNet.get_Y(inv_test_X).tolist()

    #train_Y, train_sig = net.get_Y_sigma(train_X).tolist()
    #test_Y,test_sig = net.get_Y_sigma(test_X).tolist()


    #data = [description]#data_name,train_X,train_Y, train_Y_,test_X,test_Y, test_Y_

    #inverse transform:


    #data.append([envData.denormalize_dict(envData.X_to_X_dict(train_x)) for train_x in train_X])
    #data.append([envData.denormalize_dict(envData.Y_to_Y_dict(train_y)) for train_y in train_Y])
    #data.append([envData.denormalize_dict(envData.Y_to_Y_dict(train_y_)) for train_y_ in train_Y_])
    
    #data.append([envData.denormalize_dict(envData.X_to_X_dict(x)) for x in test_X])
    #data.append([envData.denormalize_dict(envData.Y_to_Y_dict(y)) for y in test_Y])
    #data.append([envData.denormalize_dict(envData.Y_to_Y_dict(y_)) for y_ in test_Y_])

    
   # test_net_performance.save_data(os.getcwd()+"/files/train_data/"+file_name,data)
   # print("data saved")


    plt.figure(1)
    test_net_performance.plot_comparison(inv_train_Y,inv_train_Y_,"steer_command_train")
    plt.figure(2)
    test_net_performance.plot_comparison(inv_test_Y,inv_test_Y_,"steer_command_test")

    inv_filtered_test_X,inv_filtered_test_Y_ ,inv_filtered_test_Y= [],[],[]
    
    for x,y_,y in zip(inv_test_X,inv_test_Y_,inv_test_Y):
        if x[4] > 0.05:
            inv_filtered_test_X.append(x)
            inv_filtered_test_Y_.append(y_)
            inv_filtered_test_Y.append(y)
    plt.figure(3)
    #plt.plot(np.array(inv_filtered_test_X)[:,4],'o')
    test_net_performance.plot_comparison(inv_filtered_test_Y,inv_filtered_test_Y_,"steer_command_test filtered")
    plt.show()
