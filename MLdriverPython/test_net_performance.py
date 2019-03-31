
import agent_lib as pLib
import matplotlib.pyplot as plt
import numpy as np
import library as lib
#import environment1 
import json
import os
import random
#from sklearn import preprocessing
import time

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

def plot_comparison_dict_var(Y_, Y,var, feature,index = None,plot_name=" "):

    fig, ax = plt.subplots(figsize=(10,10))#name =plot_name
    real,predicted,var_list = [],[],[]

    for y_,y,v in zip(Y_,Y,var):
        y_dict_ = envData.Y_to_Y_dict(y_)
        y_dict = envData.Y_to_Y_dict(y)
        var_dict = envData.Y_to_Y_dict(v)

        real.append(y_dict_[feature])
        predicted.append(y_dict[feature])
        var_list.append(var_dict[feature])

   
    ax.plot(real,'o')
    ax.plot(predicted,'o')
    ax.errorbar(list(range(len(predicted))),predicted,yerr=np.absolute(var_list),c='r',ls='None',marker='.',ms=10,label='predicted distributions')


def convert_data(Replay,envData):
    rand_state, rand_a, rand_next_state, end,fail= map(np.array, zip(*Replay.memory))
    X,Y_ = envData.create_XY_(rand_state,rand_a,rand_next_state)
    return X,Y_,end,fail


def train_net(HP,net,vec_X,vec_Y_,envData,waitFor,num_train,separate_nets):#Replay
    losses = []
    train_count = 0
    batch_size = 64
    plt.ion()

    XandY_  = [[X,Y_] for X,Y_ in zip(vec_X,vec_Y_)]
    if separate_nets:
        fig, axes = plt.subplots(net.Y_n, 1)
        if net.Y_n == 1:
            axes = [axes]
    for i in range(num_train):
        if waitFor.stop == [True]:
            plt.close('all')
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
            loss = net.get_loss(X,Y_)
            #if loss > 10:
            #    print("what??")
            losses.append(loss)
            
            if separate_nets:
                tr_losses = np.array(losses).transpose()
                for i,tr_loss in enumerate(tr_losses):
                    axes[i].clear()
                    axes[i].plot(tr_loss)
                    axes[i].plot(lib.running_average(tr_loss,50))
            else:
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


def scale_and_split_data(scaling_type,test_part,vec_X,vec_Y_):
    if  scaling_type == "standard_scaler":#replay buffer is now already normalized, hence this does not work now
        #vec_X,vec_Y_ = envData.create_XY_1(state,a,next_state)#not normalized!
        train_X = vec_X[:int(test_part*len(vec_X))]
        train_Y_ = vec_Y_[:int(test_part*len(vec_X))]
        test_X = vec_X[int(test_part*len(vec_X)):]
        test_Y_ = vec_Y_[int(test_part*len(vec_X)):]
        scalerX = preprocessing.StandardScaler().fit(train_X)
        scalerY = preprocessing.StandardScaler().fit(train_Y_)
        print("scalerX:",scalerX.get_params())
        print("scalerY:",scalerY.get_params())
        train_X = scalerX.transform(train_X)
        train_Y_ = scalerY.transform(train_Y_)
        test_X = scalerX.transform(test_X)
        test_Y_ = scalerY.transform(test_Y_)

    else:
        #vec_X,vec_Y_ = envData.create_XY_(state,a,next_state)#normalized!
        train_X = vec_X[:int(test_part*len(vec_X))]
        train_Y_ = vec_Y_[:int(test_part*len(vec_X))]
        test_X = vec_X[int(test_part*len(vec_X)):]
        test_Y_ = vec_Y_[int(test_part*len(vec_X)):]

    return train_X,train_Y_,test_X,test_Y_

def test_net(Agent): 
    train = True
    split_buffer = True
    separate_nets = False
    variance_mode = False

    scaling_type ="scaler" # "scaler"#standard_scaler
    test_part = 0.3
    num_train = 10000000

    description = "small_state_var"
    file_name = "small_state_var.txt"
    #small_state_regular_norm_3_layers_50_nodes
    #small_state_standard_norm_3_layers_50_nodes

    #big_state_standard_norm_3_layers_50_nodes

    #big_state_standard_norm_3_layers_20_nodes_alpha_0001 - not good
    #big_state_standard_norm_4_layers_20_nodes_alpha_0001
    #big_state_standard_norm_3_layers_100_nodes saved in collect_data_test the best
    #big_state_standard_norm_3_layers_20_nodes_L2_01

    waitFor = lib.waitFor()
    #envData = environment1.OptimalVelocityPlannerData('model_based')

    #if separate_nets:
    #    from model_based_net_sep import model_based_network
    #else:
    #    from model_based_net import model_based_network
        

    #net = model_based_network(envData.X_n,envData.Y_n,HP.alpha)


    
    #velocity, steering angle, steering action, acceleration action,  rel x, rel y, rel ang, velocity next, steering next, roll
    if not Agent.HP.restore_flag:#restore anyway
        Agent.Replay.restore(Agent.HP.restore_file_path)
    #Replay.memory = Replay.memory[:10000]
    print("lenght of buffer: ",len(Agent.Replay.memory))
    if Agent.HP.restore_flag:
        Agent.nets.restore_all(Agent.HP.restore_file_path)
    
    full_replay_memory = Agent.Replay.memory
    train_replay_memory = full_replay_memory[int(test_part*len(full_replay_memory)):]
    test_replay_memory = full_replay_memory[:int(test_part*len(full_replay_memory))]

    Agent.Replay.memory = train_replay_memory#replace the replay memory with the train part


   


    #vec_X, vec_Y_, end,_ = map(list, zip(*Replay.memory))
    
    #train_X,train_Y_,test_X,test_Y_ = scale_and_split_data(scaling_type,test_part,vec_X,vec_Y_)



    
    if train:
        Agent.start_training()

        while waitFor.stop == [False]:
            Agent.trainShared.algorithmIsIn.clear()#indicates that are ready to take the lock
            with Agent.trainShared.Lock:
                Agent.trainShared.algorithmIsIn.set()
            time.sleep(1)

        Agent.stop_training()

    Agent.Replay.memory = full_replay_memory
    Agent.save()

    replay_memory = test_replay_memory
    
    TransNet_X,TransNet_Y_ = [],[]
    AccNet_X,AccNet_Y_ = [],[]
    SteerNet_X,SteerNet_Y_ = [],[]

    for ind in range(len(replay_memory)-1):
        if replay_memory[ind][3] == True or replay_memory[ind+1][4]: # done flag
            continue
        vehicle_state = replay_memory[ind][0]
        action = replay_memory[ind][2]
        vehicle_state_next = replay_memory[ind+1][0]
        rel_pos = replay_memory[ind+1][1]


        TransNet_X.append(vehicle_state+action)
        TransNet_Y_.append([vehicle_state_next[i] - vehicle_state[i] for i in range(len(vehicle_state_next))] +rel_pos)

        SteerNet_X.append(vehicle_state+[action[0],vehicle_state_next[Agent.trainHP.vehicle_ind_data['roll']]])
        SteerNet_Y_.append([action[1]])

    

    #print(Agent.nets.SteerNet.evaluate(np.array(SteerNet_X),np.array(SteerNet_Y_)))
    #SteerNet_Y = Agent.nets.SteerNet.predict(np.array(SteerNet_X))
    #plot_comparison(np.array(SteerNet_Y_)[:,0], np.array(SteerNet_Y)[:,0],"steer action")

    #print(Agent.nets.TransNet.evaluate(np.array(TransNet_X),np.array(TransNet_Y_)))

    TransNet_Y = Agent.nets.TransNet.predict(np.array(TransNet_X))

    ##vehicle_state_vec,action_vec,vehicle_state_next_vec,rel_pos_vec,vehicle_state_next_pred_vec,rel_pos_pred_vec = [],[],[],[],[],[]
    ##for x,y_,y in zip(TransNet_X,TransNet_Y_,TransNet_Y):
    ##    vehicle_state_vec.append( x[:len(Agent.trainHP.vehicle_ind_data)])
    ##    action_vec.append(x[len(Agent.trainHP.vehicle_ind_data):])
    ##    vehicle_state_next = y_[:len(Agent.trainHP.vehicle_ind_data)]
    ##    vehicle_state_next_vec.append([vehicle_state_next[i] + vehicle_state_vec[-1][i] for i in range(len(vehicle_state_next))])
    ##    rel_pos_vec.append(y_[len(Agent.trainHP.vehicle_ind_data):])
    ##    vehicle_state_next_pred = y[:len(Agent.trainHP.vehicle_ind_data)]
    ##    vehicle_state_next_pred_vec.append([vehicle_state_next_pred[i] + vehicle_state_vec[-1][i] for i in range(len(vehicle_state_next))])
    ##    rel_pos_pred_vec.append(y[len(Agent.trainHP.vehicle_ind_data):])
    

    ##for feature,ind in Agent.trainHP.vehicle_ind_data.items():
    ##    plot_comparison(np.array(vehicle_state_next_vec)[:,ind], np.array(vehicle_state_next_pred_vec)[:,ind],feature)

    ##for ind,feature in enumerate(["dx","dy","dang"]):
    ##    plot_comparison(np.array(rel_pos_vec)[:,ind], np.array(rel_pos_pred_vec)[:,ind],feature)

    plt.show()
    #train_X,train_Y_,_,_ = convert_data(ReplayTrain,envData)

    #print("test loss: ",net.get_loss(test_X,test_Y_))
    #print("train loss: ",net.get_loss(train_X,train_Y_))


    #if not variance_mode:
        #train_Y = net.get_Y(train_X).tolist()
        #test_Y = net.get_Y(test_X).tolist()

        #train_Y, train_sig = net.get_Y_sigma(train_X).tolist()
        #test_Y,test_sig = net.get_Y_sigma(test_X).tolist()


        #data = [description]#data_name,train_X,train_Y, train_Y_,test_X,test_Y, test_Y_



        #data.append([envData.denormalize_dict(envData.X_to_X_dict(train_x)) for train_x in train_X])
        #data.append([envData.denormalize_dict(envData.Y_to_Y_dict(train_y)) for train_y in train_Y])
        #data.append([envData.denormalize_dict(envData.Y_to_Y_dict(train_y_)) for train_y_ in train_Y_])
    
        #data.append([envData.denormalize_dict(envData.X_to_X_dict(x)) for x in test_X])
        #data.append([envData.denormalize_dict(envData.Y_to_Y_dict(y)) for y in test_Y])
        #data.append([envData.denormalize_dict(envData.Y_to_Y_dict(y_)) for y_ in test_Y_])

    
        #save_data(os.getcwd()+"\\files\\train_data\\"+file_name,data)
        #print("data saved")

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

        #for name in envData.Y_names:
        #    if envData.features_numbers[name] == 1:
        #        plot_distribution_dict(test_Y_,test_Y,name,plot_name = name)
        #    else:
        #        for i in range(envData.features_numbers[name]):
        #            plot_distribution_dict(test_Y_,test_Y,name,i,plot_name = name+str(i))

        #for name in envData.Y_names:
        #    plot_comparison_dict(test_Y_,test_Y,name,plot_name = name)


        ##train_Y_ = train_Y_[100:1100]
        ##train_Y = train_Y[100:1100]

        #for name in envData.Y_names:
        #    if envData.features_numbers[name] == 1:
        #        plot_distribution_dict(train_Y_,train_Y,name,plot_name = "train"+name)
        #    else:
        #        for i in range(envData.features_numbers[name]):
        #            plot_distribution_dict(train_Y_,train_Y,name,i,plot_name = "train"+name+str(i))

        #for name in envData.Y_names:
        #    if envData.features_numbers[name] == 1:
        #        plot_comparison_dict(train_Y_,train_Y,name,plot_name = "train"+name)
        #    else:
        #        for i in range(envData.features_numbers[name]):
        #            plot_comparison_dict(train_Y_,train_Y,name,i,plot_name = "train"+name+str(i))



        #plt.show()

    #else:#variance_mode
    #    train_Y,train_var = net.get_Y_and_var(train_X)
    #    test_Y,test_var = net.get_Y_and_var(test_X)

    #    #denormalize:
    #    #data.append([envData.denormalize_dict(envData.X_to_X_dict(train_x)) for train_x in train_X])
    #    #data.append([envData.denormalize_dict(envData.Y_to_Y_dict(train_y)) for train_y in train_Y])
    #    #data.append([envData.denormalize_dict(envData.Y_to_Y_dict(train_y_)) for train_y_ in train_Y_])
    
    #    #data.append([envData.denormalize_dict(envData.X_to_X_dict(x)) for x in test_X])
    #    #data.append([envData.denormalize_dict(envData.Y_to_Y_dict(y)) for y in test_Y])
    #    #data.append([envData.denormalize_dict(envData.Y_to_Y_dict(y_)) for y_ in test_Y_])


        
    #    for name in envData.Y_names:
    #        plot_comparison_dict_var(test_Y_,test_Y,train_var,name,plot_name = name)
    #    plt.show()