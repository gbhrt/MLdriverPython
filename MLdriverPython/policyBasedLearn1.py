
from planner import Planner
import numpy as np
from library import *  #temp
from classes import Path
import copy
import random
from policyBasedNet import Network
from plot import Plot
import json
import policyBasedLib as pLib
import data_manager

if __name__ == "__main__": 
    #run simulator: cd C:\Users\student_2\Documents\ArielUnity - learning2\sim_2_1
    # cd C:\Users\gavri\Desktop\sim_2_1
    #sim_2_1_17 -quit -batchmode -nographics

    #pre-defined parameters:
    feature_points = 10 #not neccecery at the begining also 1 is good
    distance_between_points = 1.0 #meter
    features_num = feature_points #vehicle state points on the path (distance)
    #epsilon = 0.2
    gamma = 0.99
    num_of_runs = 5000
    step_time = 0.2#0.02
    alpha_actor = 0.00001# for Pi 1e-5 #learning rate
    alpha_critic = 0.00001#for Q
    #max_deviation = 3 # [m] if more then maximum - end episode 
    batch_size = 1
    num_of_TD_steps = 15 #for TD(lambda)
    visualized_points = 300 #how many points show on the map - just visualy
    max_pitch = 0.3
    max_roll = 0.3
    acc = 1.5 # [m/s^2]  need to be more then maximum acceleration in real
    res = 1
    plot_flag = True
    restore_flag = True
    skip_run = False
    random_paths_flag = True
    reset_every = 3
    save_every = 25
    path_name = "paths\\‏‏straight_path_limit3.txt"     #long random path: path3.txt  #long straight path: straight_path.txt
    save_name = "model14" #model6.ckpt - constant velocity limit - good. model7.ckpt - relative velocity.
    #model10.ckpt TD(5) dt = 0.2 alpha 0.001 model13.ckpt - 5 points 2.5 m 0.001 TD 15
    #model8.ckpt - offline trained after 5 episode - very clear
    restore_name = "model14" # model2.ckpt - MC estimation 
    run_data_file_name = 'running_record1'

    ###################
    value_vec_tot = []
    state_vec_tot= []
    action_vec_tot= []
    steps = [0,0]#for random action selection of random number of steps
    stop = []
    wait_for(stop)#wait for "enter" in another thread - then stop = true
    dv = acc * step_time / res
    action_space =[-dv,dv]
    net = Network(features_num,len(action_space),alpha_actor,alpha_critic)   
    print("Network ready")
    if restore_flag:
        net.restore(restore_name)
    if not skip_run:
        pl = Planner()
        pl.start_simple()
        if(pl.load_path(path_name,random_paths_flag)):
            stop[0] = 1

        last_state = [0 for _ in range(batch_size)]
        Q_ = [0 for _ in range(batch_size)]

        plot = Plot()
        dataManager = data_manager.DataManager()

        for i in range(num_of_runs): #number of runs - run end at the end of the main path and if vehicle deviation error is to big
            if stop:
                break
            # initialize every episode:
            last_time = [0]
            batch_index= 0 #first time last Q and last state not exist,  add to batch just from next step
            TD_index = 0
            count = 0
            total_reward = 0
            state_vec = []
            action_vec = []
            value_vec = []
            reward_vec = [] #save lambda rewards
            #########################
            pl.simulator.get_vehicle_data()#read data after time step from last action
            if random_paths_flag:
                if(pl.load_path(path_name,random_paths_flag)):#if cannot load path
                    break
                
            pl.new_episode()#compute path in current vehicle position
            #first step:
            local_path = pl.get_local_path()#num_of_points = visualized_points
            state = pLib.get_state(pl,local_path,feature_points,distance_between_points)
            Q = net.get_Q(state)
            Pi = net.get_Pi(state)
            #make action: 
            a,one_hot_a = pLib.choose_action(action_space,Pi,steps)#choose action 
            #a,one_hot_a = pLib..choose_action(action_space,Q,steps)
            pl.delta_velocity_command(action_space[a])#update velocity (and steering) and send to simulator. index - index on global path (pl.desired_path)
            dataManager.update_real_path(pl = pl,velocity_limit = local_path.velocity_limit[0])
        

            while not stop:#while not stoped, the loop break if reached the end or the deviation is to big
                if step_now(last_time,step_time):#check if make the next step (after step_time) 
                    pl.simulator.get_vehicle_data()#read data after time step from last action
                    #print("angle: ",pl.simulator.vehicle.angle)
                    local_path = pl.get_local_path(send_path = False,num_of_points = visualized_points)#num_of_points = visualized_points
                    last_state[batch_index] =copy.copy(state)#copy current state to list of last states
                    state = pLib.get_state(pl,local_path,feature_points,distance_between_points)

                    reward = pLib.get_reward(last_state[batch_index],state,local_path.velocity_limit[0])
                    Q_[batch_index] = np.copy(Q)#save Q from last state
               
                  #  print("__________________________________________")
                #print("deviation: ",state[1]," steering: ",state[0],"x: ",state[2]," y: ",state[3]," reward: ",reward)#
                    
                    #print("velocity",state[0]," reward: ",reward)#,"distance: ",state[1],
                    #W1,b1 = net.get_par()
                    #print("W0 - before: ",W1)
                    #print("b - before: ",b1)
                   # print("Q - before: ",Q)

                    #actor-critic:
                    Q = net.get_Q(state)#from this state
                    #print("correcting with action: ",a)
                    Q_[batch_index][a] = reward +gamma*np.max(Q)#compute Q_: reward,a - from last to this state, Q - from this state
                    net.Update_Q(last_state,Q_)

                    Q_last = net.get_Q(last_state[0])#Q for last state for update policy on last state values
                    Q_loss = net.get_Q_loss(last_state[0],Q_[batch_index])
                    print("Q_loss: ",Q_loss[0][a])
                    A = Q_last[a] - sum(Q_last)/len(Q_last)
                    net.Update_policy(last_state,[one_hot_a],[[A]])

                    #print("Q_: ",Q_[batch_index])
                    #loss = net.get_value_loss(last_state[batch_index],Q_[batch_index])
                    # print("loss: ",loss[0][a])    
                    #W1,b1 = net.get_par()
                    #print("W0 - after: ",W1)
                    #print("b - after: ",b1)
                
                    #Q_last = net.get_Q(last_state)#temp
                    #print ("last_state: ",last_state)
                    #print("Q - after update (last step):",Q_last)
                    #print("Q - after:",Q)
                

                    batch_index += 1
                    if batch_index >= batch_size:      
                        ##A = Q[a] - sum(Q) / float(len(Q))
                        #Pi = net.get_Pi(last_state[0])
                        #print("PI before: ",Pi)
                        #net.Update_policy(last_state,[one_hot_a],[[reward]])#Q[a])

                        #net.Update_value(last_state,Q_)#update session on the mini-batch
                        batch_index = 0

                        #Pi = net.get_Pi(last_state[0])
                        #print("PI after: ",Pi)
                 
                    #print ("state: ",state)
                    
                    
                    #TD(lambda):
                    ##if(len(state_vec) > num_of_TD_steps):
                    ##    future_reward = 0
                    ##    for k in range(num_of_TD_steps):
                    ##        future_reward += reward_vec[k]*gamma**k 
                    ##    net.Update_policy([state_vec[-num_of_TD_steps]],[action_vec[-num_of_TD_steps]],[[future_reward]])
                    ##    reward_vec.pop(0)#remove first
                    ##    value_vec[-num_of_TD_steps] = [future_reward]#the reward is on the last state and action
                    ##reward_vec.append(reward)

                    ##state_vec.append(last_state[0])#for one batch only
                    ##action_vec.append(one_hot_a)
                    value_vec.append([reward])#the reward is on the last state and action

                
                    #make action:
                    Pi = net.get_Pi(state)
                    print("velocity1: ",state[0],"Q: ",Q,"PI: ",Pi)#"velocity2: ",state[1],
                    a,one_hot_a = pLib.choose_action(action_space,Pi,steps)#choose action 
                    #a,one_hot_a = pLib.choose_action(action_space,Q,steps)
                    pl.delta_velocity_command(action_space[a])#update velocity (and steering) and send to simulator. index - index on global path (pl.desired_path)        
                    dataManager.update_real_path(pl = pl,velocity_limit = local_path.velocity_limit[0])#state[0]
                    dataManager.save_additional_data(pl,features = state,action = a)
                    dev = dist(local_path.position[0][0],local_path.position[0][1],0,0)
                    mode = pl.check_end(deviation = dev)#check if end of the episode 
                    if mode != 'ok':
                        break
                
                #end if time
            #end while
            #after episode end:
            #net.save_model()
            #time.sleep(1)
            if mode != 'kipp':
                pl.stop_vehicle()

            if (i % reset_every == 0 and i > 0) or mode == 'kipp': 
                #pl.stop_vehicle()
                pl.simulator.reset_position()
                pl.stop_vehicle()
                #net.save_model()
            if (i % save_every == 0 and i > 0): 
                net.save_model(save_name)
            reward_sum = 0
            for item in value_vec: reward_sum+=item[0]
            print("mean reward: ",reward_sum/len(value_vec))
            if plot_flag:
                plot.close()
                plot.plot_path_with_features(dataManager,distance_between_points)
                #plot.plot_path(dataManager.real_path)
            pl.restart()#stop vehicle, and initalize real path
            dataManager.restart()
           
            #comp_Pi(net)
            #update policy at the end of the episode:

            #for i in range(len(value_vec)-2, -1, -1):
            #    value_vec[i] += gamma * value_vec[i+1]
            #value_vec = [[x] for x in value_vec]

            #value_vec_tot += value_vec
            #state_vec_tot += state_vec
            #action_vec_tot += action_vec
            #for i in range(1000000):
            #    if stop:
            #        break
            #    #for i in range(len(state_vec)-1):
            #    net.Update_policy(state_vec,action_vec,value_vec)
            #    loss = net.get_policy_loss(state_vec,action_vec,value_vec)
            #    if i%100 == 0:
            #        #loss = sum(loss) / float(len(loss))
            #        #print("state_vec: ",state_vec)
            #        comp_value(net,max_velocity)
            #        print("loss: ",loss) 
                    #net.update_sess(state_vec[i],action_vec[i],value_vec[i])#update session on the mini-batch               
                    #net.Update_value(state_vec[i],value_vec[i])#update session on the mini-batch
 
    
           # net.update_sess(last_state,Q_corrected)#update session on the mini-batch
        #end for num_of_runs
        pl.stop_vehicle()


        #with open(run_data_file_name, 'w') as f:
        #    json.dump([state_vec_tot,action_vec_tot,value_vec_tot], f)


        pl.end()
    #else:#run skiped
    #    with open(run_data_file_name, 'r') as f:
    #        [state_vec_tot,action_vec_tot,value_vec_tot] = json.load(f)
    #for i in range(len(state_vec_tot)):
    #    print("s: ",state_vec_tot[i], "a: ",action_vec_tot[i],"r:",value_vec_tot[i])
    #for i in range(1000000):
    #    if stop:
    #        break
    #    for j in range(len(state_vec_tot)):#stochastic update
    #        net.Update_policy([state_vec_tot[j]],[action_vec_tot[j]],[value_vec_tot[j]])#Q[a])
    #    #net.Update_value(state_vec_tot,value_vec_tot)
    #    #action_vec.pop(0)#remove first value
        
    #    if i%1 == 0:
    #        loss = net.get_policy_loss(state_vec_tot,action_vec_tot,value_vec_tot)
    #        #loss = sum(loss) / float(len(loss))
    #        #print("state_vec: ",state_vec)
    #        comp_Pi(net)
    #        print("loss: ",loss) 
            
    net.save_model(save_name)#model4.ckpt - LINEAR, LINE. model5.ckpt - net, line - good, model6.ckpt - 3 points model7.ckpt -very good
    
