from model_based_net import model_based_network
import agent_lib as pLib
import hyper_parameters
import matplotlib.pyplot as plt
import numpy as np
import library as lib
import enviroment1 
import json
import os
import random
from sklearn import preprocessing
def save_data(file_name,data):
    with open(file_name, 'w') as f:#append data to the file
        json.dump(data,f)

def plot_distribution(data,name):
    plt.figure(name+" distribution")
    plt.hist(data,bins='auto')
    print("variance of",name,":",np.var(data))
def plot_comparison(real, predicted,name):
    plt.figure(name)
    plt.plot(real,'o')
    plt.plot(predicted,'o')

def plot_distribution_dict(Y_, Y,feature,index = None,plot_name=" "):
    plt.figure(plot_name+" distribution")
    real,predicted = [],[]
    for y_,y in zip(Y_,Y):
        y_dict_ = envData.Y_to_Y_dict(y_)
        y_dict = envData.Y_to_Y_dict(y)
        if index is None:
            real.append(y_dict_[feature])
            predicted.append(y_dict[feature])
        else:
            real.append(y_dict_[feature][index])
            predicted.append(y_dict[feature][index])
    error = np.array(real) - np.array(predicted)
    plt.hist(error,bins='auto')
    print("variance of",name,":",np.var(error))

def plot_comparison_dict(Y_, Y,feature,index = None,plot_name=" "):
    plt.figure(plot_name)
    real,predicted = [],[]
    for y_,y in zip(Y_,Y):
        y_dict_ = envData.Y_to_Y_dict(y_)
        y_dict = envData.Y_to_Y_dict(y)
        if index is None:
            real.append(y_dict_[feature])
            predicted.append(y_dict[feature])
        else:
            real.append(y_dict_[feature][index])
            predicted.append(y_dict[feature][index])

   
    plt.plot(real,'o')
    plt.plot(predicted,'o')




def convert_data(Replay,envData):
    rand_state, rand_a, rand_next_state, end,fail= map(np.array, zip(*Replay.memory))
    X,Y_ = envData.create_XY_(rand_state,rand_a,rand_next_state)
    return X,Y_,end,fail


def train_net(HP,net,vec_X,vec_Y_,envData,waitFor,num_train):#Replay
    losses = []
    train_count = 0
    batch_size = 64
    plt.ion()

    XandY_  = [[X,Y_] for X,Y_ in zip(vec_X,vec_Y_)]
    for i in range(num_train):
        if waitFor.stop == [True]:
            break
        #with trainShared.Lock:
        #rand_state, rand_a, rand_next_state, rand_end,_ = Replay.sample(HP.batch_size)

        #X,Y_ = map(list, zip(*random.sample(zip(vec_X,vec_Y_),np.clip(batch_size,0,len(vec_Y_)))))
       # X,Y_ = map(list, zip(*random.sample(XandY_,np.clip(batch_size,0,len(XandY_)))))
        X,Y_ = zip(*random.sample(XandY_,np.clip(batch_size,0,len(XandY_))))

        #update neural networs:
        #pLib.model_based_update(rand_state, rand_a, rand_next_state,rand_end,net,HP,envData)
        #X,Y_ = envData.create_XY_(rand_state, rand_a, rand_next_state)
        net.update_network(X,Y_)
        train_count+=1
        if train_count % 500 == 0:
            #X,Y_,end,fail = convert_data(ReplayTest)
            loss = float(net.get_loss(X,Y_))
            if loss > 10:
                print("what??")
            losses.append(loss)
            plt.cla()
            plt.plot(losses)
            plt.plot(lib.running_average(losses,50))
            plt.draw()
            plt.pause(0.0001)
            print("train:",train_count, "loss:",loss)
    plt.ioff()        
    plt.show()
    net.save_model(HP.save_file_path)

def comp_abs_states(init_sample,pred_vec):
    pos = [0,0]
    ang = 0
    roll = 0
    states = [[pos[0],pos[1],ang,init_sample[0],init_sample[1],roll]]#pos ang vel steer 
    for pred in pred_vec:
        pos = lib.to_global(pred[:2], pos, ang)
        ang = pred[2] + ang#
        states.append([pos[0],pos[1],ang,pred[3],pred[4],pred[5]])
        
    return states

def predict_n_next(net,X,end,n):
    pred_vec = []
    sample = list(X[0])
    if end[0] == True:
        return pred_vec
    for i in range(1,n):#n times 
        if end[i] == True:
            break
        #print(sample)
        prediction = net.get_Y([sample])[0]#predict_next(features_num,train_data, sample, k, p)
        pred_vec.append(prediction)#x,y,ang,vel, steer - all relative 
        sample[0] = prediction[3]#vel
        sample[1] = prediction[4]#steer
        sample[2] = X[i][2]#action steer
        sample[3] = X[i][3] #action acceleration
        #print(prediction)
    return pred_vec

def compare_n_samples(net,X,end,Y_,n):
    pred_vec = predict_n_next(net,X,end,n)
    abs_pred = comp_abs_states(X[0],pred_vec)    
    abs_real_pred = comp_abs_states(X[0],Y_[:len(pred_vec)])
    return abs_pred,abs_real_pred


if __name__ == "__main__": 
    restore = True
    train = False
    split_buffer = True

    scaling_type = "scaler"#standard_scaler
    test_part = 0.3
    num_train = 10000000

    description = "small_state_standard_norm_3_layers_100_nodes_L2_01"
    file_name = "small_state_standard_norm_3_layers_100_nodes_L2_01.txt"
    #small_state_regular_norm_3_layers_50_nodes
    #small_state_standard_norm_3_layers_50_nodes

    #big_state_standard_norm_3_layers_50_nodes

    #big_state_standard_norm_3_layers_20_nodes_alpha_0001 - not good
    #big_state_standard_norm_4_layers_20_nodes_alpha_0001
    #big_state_standard_norm_3_layers_100_nodes saved in collect_data_test the best
    #big_state_standard_norm_3_layers_20_nodes_L2_01

    waitFor = lib.waitFor()
    HP = hyper_parameters.ModelBasedHyperParameters()
    envData = enviroment1.OptimalVelocityPlannerData('model_based')
    net = model_based_network(envData.X_n,envData.Y_n,HP.alpha,envData.observation_space.range)

    Replay = pLib.Replay(1000000)
    #velocity, steering angle, steering action, acceleration action,  rel x, rel y, rel ang, velocity next, steering next, roll
    Replay.restore(HP.restore_file_path)
    #Replay.memory = Replay.memory[:10000]
    print("lenght of buffer: ",len(Replay.memory))

    state, a, next_state, end,_ = map(list, zip(*Replay.memory))#all data
    if  scaling_type == "standard_scaler":
        vec_X,vec_Y_ = envData.create_XY_1(state,a,next_state)#not normalized!
        train_X = vec_X[:int(test_part*len(vec_X))]
        train_Y_ = vec_Y_[:int(test_part*len(vec_X))]
        test_X = vec_X[int(test_part*len(vec_X)):]
        test_Y_ = vec_Y_[int(test_part*len(vec_X)):]
        scalerX = preprocessing.StandardScaler().fit(train_X)
        scalerY = preprocessing.StandardScaler().fit(train_Y_)
        train_X = scalerX.transform(train_X)
        train_Y_ = scalerY.transform(train_Y_)
        test_X = scalerX.transform(test_X)
        test_Y_ = scalerY.transform(test_Y_)

    else:
        vec_X,vec_Y_ = envData.create_XY_(state,a,next_state)#normalized!
        train_X = vec_X[:int(test_part*len(vec_X))]
        train_Y_ = vec_Y_[:int(test_part*len(vec_X))]
        test_X = vec_X[int(test_part*len(vec_X)):]
        test_Y_ = vec_Y_[int(test_part*len(vec_X)):]

    if restore:
        net.restore(HP.restore_file_path)

    
    if train:
        train_net(HP,net,train_X,train_Y_,envData, waitFor,num_train)#ReplayTrain

    
    
    #train_X,train_Y_,_,_ = convert_data(ReplayTrain,envData)
    train_Y = net.get_Y(train_X).tolist()
    test_Y = net.get_Y(test_X).tolist()
    print("test loss: ",net.get_loss(test_X,test_Y_))
    print("train loss: ",net.get_loss(train_X,train_Y_))
    data = [description]#data_name,train_X,train_Y, train_Y_,test_X,test_Y, test_Y_

    #inverse transform:
    if scaling_type == "standard_scaler":
        train_Y = scalerY.inverse_transform(train_Y)
        train_Y_ = scalerY.inverse_transform(train_Y_)
        train_X = scalerX.inverse_transform(train_X)

        test_Y = scalerY.inverse_transform(test_Y)
        test_Y_ = scalerY.inverse_transform(test_Y_)
        test_X = scalerX.inverse_transform(test_X)

        data.append([envData.X_to_X_dict(train_x) for train_x in train_X])
        data.append([envData.Y_to_Y_dict(train_y) for train_y in train_Y])
        data.append([envData.Y_to_Y_dict(train_y_) for train_y_ in train_Y_])
    
        data.append([envData.X_to_X_dict(x) for x in test_X])
        data.append([envData.Y_to_Y_dict(y) for y in test_Y])
        data.append([envData.Y_to_Y_dict(y_) for y_ in test_Y_])
    else:

        data.append([envData.denormalize_dict(envData.X_to_X_dict(train_x)) for train_x in train_X])
        data.append([envData.denormalize_dict(envData.Y_to_Y_dict(train_y)) for train_y in train_Y])
        data.append([envData.denormalize_dict(envData.Y_to_Y_dict(train_y_)) for train_y_ in train_Y_])
    
        data.append([envData.denormalize_dict(envData.X_to_X_dict(x)) for x in test_X])
        data.append([envData.denormalize_dict(envData.Y_to_Y_dict(y)) for y in test_Y])
        data.append([envData.denormalize_dict(envData.Y_to_Y_dict(y_)) for y_ in test_Y_])
    #compare one step prediction:
    #Y = np.array(Y)
    #Y_ = np.array(Y_)
    #errors = Y - Y_

    

    save_data(os.getcwd()+"\\files\\train_data\\"+file_name,data)
    print("data saved")

    ##compare n step prediction:
    #vec_Y_n =  []
    #vec_Y_n_ = []
    #n = 2
    #for i in range(len(X)-n):
    #    Y_n, Y_n_ = compare_n_samples(net,X[i:i+n],end[i:i+n],Y_[i:i+n],n)#compute the predicted and real state during n steps 
    #    if  len(Y_n) < n:
    #        continue
    #    vec_Y_n_.append(Y_n_[-1])
    #    vec_Y_n.append(Y_n[-1])

    #vec_Y_n = np.array(vec_Y_n)
    #vec_Y_n_ = np.array(vec_Y_n_)
    #errors = vec_Y_n - vec_Y_n_

    #fail_Y, fail_Y_ = [],[]
    #for y, y_ in zip(vec_Y_n,vec_Y_n_):
    #    if abs(y[5]) < envData.max_plan_roll and abs(y_[5]) > envData.max_plan_roll:
    #        fail_Y.append(y[5])
    #        fail_Y_.append(y_[5])
    #total_fails = 0
    #for roll in vec_Y_n_[:,5]:
    #    if roll > envData.max_plan_roll:
    #        total_fails+=1 
    #print("total_fails:",total_fails)
    #print("fails: ",len(fail_Y))
    #plot_comparison(fail_Y_, fail_Y,"fails")

    #plot_distribution(errors[:,0],"error x")
    #test_Y_ = test_Y_[100:1100]
    #test_Y = test_Y[100:1100]

    for name in envData.Y_names:
        if envData.features_numbers[name] == 1:
            plot_distribution_dict(test_Y_,test_Y,name,plot_name = name)
        else:
            for i in range(envData.features_numbers[name]):
                plot_distribution_dict(test_Y_,test_Y,name,i,plot_name = name+str(i))

    for name in envData.Y_names:
        if envData.features_numbers[name] == 1:
            plot_comparison_dict(test_Y_,test_Y,name,plot_name = name)
        else:
            for i in range(envData.features_numbers[name]):
                plot_comparison_dict(test_Y_,test_Y,name,i,plot_name = name+str(i))

    #train_Y_ = train_Y_[100:1100]
    #train_Y = train_Y[100:1100]

    for name in envData.Y_names:
        if envData.features_numbers[name] == 1:
            plot_distribution_dict(train_Y_,train_Y,name,plot_name = "train"+name)
        else:
            for i in range(envData.features_numbers[name]):
                plot_distribution_dict(train_Y_,train_Y,name,i,plot_name = "train"+name+str(i))

    for name in envData.Y_names:
        if envData.features_numbers[name] == 1:
            plot_comparison_dict(train_Y_,train_Y,name,plot_name = "train"+name)
        else:
            for i in range(envData.features_numbers[name]):
                plot_comparison_dict(train_Y_,train_Y,name,i,plot_name = "train"+name+str(i))



    plt.show()

       