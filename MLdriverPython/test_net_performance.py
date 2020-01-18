
import agent_lib as pLib
import matplotlib.pyplot as plt
import numpy as np
import library as lib
#import environment1 
import json
import os
import random
import direct_method
#from sklearn import preprocessing
import time
import agent
import agent_lib
import predict_lib
from statsmodels.graphics.gofplots import qqplot

def save_data(file_name,data):
    with open(file_name, 'w') as f:#append data to the file
        json.dump(data,f)


        
def plot_qqplot(data,name):
    qqplot(data, line='s',label = name)
    plt.legend()

def plot_distribution(data,name):
    var = np.std(data,dtype=np.float64)
    plt.figure(name+" distribution")
    plt.title(name,fontsize = 30)
    plt.tick_params(labelsize=20)
    plt.hist(data,bins='auto',range=[-3*var, 3*var])
    print("variance of",name,":",var)

def plot_comparison(real, predicted,name):
    
    plt.figure(name)
    plt.title(name)
    plt.plot(real,'o',label = "real")
    plt.plot(predicted,'o',label = "predicted")
    plt.legend()

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





#def convert_replay_to_states(replay_memory):
#    states = []
#    actions = []
#    done = []
    
#    for ind in range(len(replay_memory)-1):
#        StateVehicle = agent.State().Vehicle
#        if replay_memory[ind][3] == True or replay_memory[ind+1][4]: # done flag
#            continue
#        StateVehicle.values = replay_memory[ind][0]
#        actions.append(replay_memory[ind][2])
#        vehicle_state_next = replay_memory[ind+1][0]
#        rel_pos = replay_memory[ind+1][1]

def plot_n_step_state(Agent,replay_memory):
    waitFor = lib.waitFor()
    #n step:
    fig1,ax_abs = plt.subplots(1)
    ax_abs.axis('equal')

    fig2,axes = plt.subplots(len(Agent.trainHP.vehicle_ind_data),constrained_layout=True)
    plt.ion()
    fontsize = 15
    n = 20
    for i in range (63,64):# (len(replay_memory)-n):#:
        print("index:",i)
        if waitFor.stop == [True]:
            break
        replay_memory_short = replay_memory[i:i+n]
        vehicle_state_vec,action_vec,abs_pos_vec,abs_ang_vec = real_to_abs_n_steps(replay_memory_short)
        pred_vehicle_state_vec,pred_abs_pos_vec,pred_abs_ang_vec = predict_lib.predict_n_steps(Agent,vehicle_state_vec[0],abs_pos_vec[0],abs_ang_vec[0],action_vec)
        x,y = zip(*abs_pos_vec)
        p_x,p_y = zip(*pred_abs_pos_vec)
        #compare_states = zip(vehicle_state_vec,pred_vehicle_state_vec)
        
        fig2.suptitle('Multi-step prediction', fontsize=fontsize)
        
        for feature,ind in Agent.trainHP.vehicle_ind_data.items():
            real = np.array(vehicle_state_vec)[:,ind]
            pred = np.array(pred_vehicle_state_vec)[:,ind]
            error = pred - real
            axes[ind].clear()
            
            axes[ind].set_ylabel(feature, fontsize=fontsize)
            #x = list(range(n))
            axes[ind].plot(real,color = "red")
            axes[ind].plot(pred,color = "blue")
            axes[ind].xaxis.set_ticks(np.arange(0, n, 1))
        axes[-1].set_xlabel('Step number',fontsize=fontsize)
        fig2.legend()
        
        ax_abs.clear()
        ax_abs.set_xlabel('X', fontsize=fontsize)
        ax_abs.set_ylabel('Y', fontsize=fontsize)
        ax_abs.plot(x,y,color = "red")
        ax_abs.plot(p_x,p_y,color = "blue")
        plt.draw()
        plt.pause(0.0001)
    plt.ioff()
    plt.show()






def plot_n_step_var(Agent,replay_memory):
    max_n = 10
    n_list = list(range(2,max_n))

    n_state_vec,n_state_vec_pred,n_pos_vec,n_pos_vec_pred,n_ang_vec,n_ang_vec_pred = predict_lib.get_all_n_step_states(Agent.Direct if Agent.trainHP.direct_predict_active else Agent.nets.TransNet, Agent.trainHP,replay_memory, max_n)

    var_vec,mean_vec,pos_var_vec,pos_mean_vec,ang_var_vec,ang_mean_vec = predict_lib.comp_var(Agent, n_state_vec,n_state_vec_pred,n_pos_vec,n_pos_vec_pred,n_ang_vec,n_ang_vec_pred)

    #max_n = 12
    #var_vec = list(range(2,max_n))
    #mean_vec= list(range(2,max_n))
    #pos_var_vec= list(range(2,max_n))
    #pos_mean_vec= list(range(2,max_n))
    #ang_var_vec= list(range(2,max_n))
    #ang_mean_vec= list(range(2,max_n))

    fig2,axes = plt.subplots(len(Agent.trainHP.vehicle_ind_data)+3,constrained_layout=True)
    
    
    n_list = list(range(1,max_n))#for ploting from 1 -1
    fontsize = 20

    for ind,feature in enumerate([r'$\Delta x[m]$',r'$\Delta y[m]$']): 
        axes[ind+len(Agent.trainHP.vehicle_ind_data)].set_ylabel(feature, fontsize=fontsize)
        var = np.array(pos_var_vec)[:,ind]
        mean = np.array(pos_mean_vec)[:,ind]     
        #axes[ind+len(Agent.trainHP.vehicle_ind_data)].plot(n_list,var)
        axes[ind+len(Agent.trainHP.vehicle_ind_data)].tick_params(labelsize=15)
        axes[ind+len(Agent.trainHP.vehicle_ind_data)].errorbar(n_list,mean,alpha = 0.7)#var
        axes[ind+len(Agent.trainHP.vehicle_ind_data)].fill_between(n_list,mean+var,mean-var,color = "#dddddd" )

    axes[len(Agent.trainHP.vehicle_ind_data)+2].set_ylabel(r'$\Delta \theta_z[rad]$', fontsize=fontsize)
    var = np.array(ang_var_vec)
    mean = np.array(ang_mean_vec) 
    #axes[len(Agent.trainHP.vehicle_ind_data)+2].plot(n_list,var)
    axes[len(Agent.trainHP.vehicle_ind_data)+2].tick_params(labelsize=15)
    axes[len(Agent.trainHP.vehicle_ind_data)+2].errorbar(n_list,mean,alpha = 0.7)#,var
    axes[len(Agent.trainHP.vehicle_ind_data)+2].fill_between(n_list,mean+var,mean-var,color = "#dddddd" )

    #for ind,feature in enumerate([r'$v[m/s]$',r'$\delta[rad]$',r'$\theta_y[rad]$']): 
    for ind,feature in enumerate(Agent.trainHP.vehicle_ind_data.keys()): 
        axes[ind].set_ylabel(feature, fontsize=fontsize)
        var = np.array(var_vec)[:,ind]
        mean = np.array(mean_vec)[:,ind]
        #axes[ind].plot(n_list,var)
        axes[ind].tick_params(labelsize=15)
        axes[ind].errorbar(n_list,mean,alpha = 0.7)#,var
        axes[ind].fill_between(n_list,mean+var,mean-var,color = "#dddddd" )

    axes[-1].set_xlabel('Step number',fontsize=fontsize)
    #fig2.legend()
    plt.show()



def one_step_pred_plot(Agent,replay_memory):  

    vehicle_state_next_vec,vehicle_state_next_pred_vec,rel_pos_vec,rel_pos_pred_vec = predict_lib.one_step_prediction(Agent,replay_memory)
        

    for feature,ind in Agent.trainHP.vehicle_ind_data.items():
        plot_comparison(np.array(vehicle_state_next_vec)[:,ind], np.array(vehicle_state_next_pred_vec)[:,ind],feature)
        

    for ind,feature in enumerate([r'$\Delta x$',r'$\Delta y$',r'$\Delta \theta_z$']):
        real = np.array(rel_pos_vec)[:,ind]
        pred = np.array(rel_pos_pred_vec)[:,ind]
        error = pred - real
        print("mse:",feature,np.sqrt((error**2).mean()))
        plot_qqplot(error,feature)
        plot_distribution(error,feature)
        plot_comparison(real, pred,feature)


    # for ind,feature in enumerate([r'$v$',r'$\delta$',r'$\theta_y$']):
    for ind,feature in enumerate([r'$v$',r'$\delta$']):

        real = np.array(vehicle_state_next_vec)[:,ind]
        pred = np.array(vehicle_state_next_pred_vec)[:,ind]
        error = pred - real
        print("mse:",feature,np.sqrt((error**2).mean()))
        plot_distribution(error,feature)
        plot_qqplot(error,feature)
    #LTR
    steer_vec_real = np.array(vehicle_state_next_vec)[:,Agent.trainHP.vehicle_ind_data["steer"]]
    vel_vec_real = np.array(vehicle_state_next_vec)[:,Agent.trainHP.vehicle_ind_data["vel_y"]]
    LTR_vec_real = [Agent.Direct.comp_LTR(vel,steer) for steer, vel in zip(steer_vec_real,vel_vec_real)]
    steer_vec_pred = np.array(vehicle_state_next_pred_vec)[:,Agent.trainHP.vehicle_ind_data["steer"]]
    vel_vec_pred = np.array(vehicle_state_next_pred_vec)[:,Agent.trainHP.vehicle_ind_data["vel_y"]]
    LTR_vec_pred = [Agent.Direct.comp_LTR(vel,steer) for steer, vel in zip(steer_vec_pred,vel_vec_pred)]
    error = np.array(LTR_vec_pred) - np.array(LTR_vec_real)
    plot_qqplot(error,"LTR")
    plot_distribution(error,"LTR")
    plot_comparison(LTR_vec_real, LTR_vec_pred,"LTR")

    var = Agent.planningState.var[1]
    num = 0
    for e in error:
        if abs(e) > var:
            num+=1

    print("percent of deviation from saftey margin:",num)    
    plt.show()

def train_nets(Agent):
    waitFor = lib.waitFor()
    Agent.start_training()
    while waitFor.stop == [False]:
        Agent.trainShared.algorithmIsIn.clear()#indicates that are ready to take the lock
        with Agent.trainShared.Lock:
            Agent.trainShared.algorithmIsIn.set()
        time.sleep(1)
    Agent.stop_training()


#def direct():

def test_net(Agent): 
    train = False
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
        Agent.nets.restore_all(Agent.HP.restore_file_path,Agent.HP.net_name)
    
    full_replay_memory = Agent.Replay.memory
    train_replay_memory = full_replay_memory[int(test_part*len(full_replay_memory)):]
    test_replay_memory = full_replay_memory[:int(test_part*len(full_replay_memory))]


    Agent.Replay.memory = train_replay_memory#replace the replay memory with the train part


   


    #vec_X, vec_Y_, end,_ = map(list, zip(*Replay.memory))
    
    #train_X,train_Y_,test_X,test_Y_ = scale_and_split_data(scaling_type,test_part,vec_X,vec_Y_)


    if train:
       train_nets(Agent)

    Agent.Replay.memory = full_replay_memory
    
    Agent.save()

    replay_memory = full_replay_memory#full_replay_memory[:int(0.3*len(full_replay_memory))]#test_replay_memory# #test_replay_memory#test_replay_memory

    #Agent.update_episode_var()#len(replay_memory)
    #TransNet_X = [[1,1,1,1,1]]

    #TransNet_Y = Agent.nets.TransNet.predict(np.array(TransNet_X))[0]
    #print(TransNet_Y)
    
    
    #plot_n_step_state(Agent,replay_memory)


    #plot_n_step_var(Agent,replay_memory)
    
    one_step_pred_plot(Agent,replay_memory)

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

    
        #save_data(os.getcwd()+"/files/train_data/"+file_name,data)
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