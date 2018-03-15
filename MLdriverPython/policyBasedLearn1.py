
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
    HP = pLib.HyperParameters()
    ###################

    Replay = pLib.Replay(HP.replay_memory_size)
    value_vec_tot = []
    state_vec_tot= []
    action_vec_tot= []
    steps = [0,0]#for random action selection of random number of steps
    stop = []
    wait_for(stop)#wait for "enter" in another thread - then stop = true
    dv = HP.acc * HP.step_time / HP.res
    action_space =[-dv,dv]
    net = Network(HP.features_num,len(action_space),HP.alpha_actor,HP.alpha_critic,tau = HP.tau)   
    print("Network ready")
    if HP.restore_flag:
        net.restore(HP.restore_name)
    if not HP.skip_run:
        pl = Planner()
        pl.start_simple()
        if not HP.random_paths_flag:
            if pl.load_path(HP.path_name,HP.random_paths_flag) == -1:
                stop = [1]

        
        vec_state = []
        vecQ_ = []
      
        plot = Plot()
        dataManager = data_manager.DataManager()

        for i in range(HP.num_of_runs): #number of runs - run end at the end of the main path and if vehicle deviation error is to big
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
            if HP.random_paths_flag:
                path_num = pl.load_path(HP.path_name,HP.random_paths_flag)
                if path_num == -1:#if cannot load path
                    break
                
            pl.new_episode()#compute path in current vehicle position
            #first state:
            local_path = pl.get_local_path()#num_of_points = HP.visualized_points
            state = pLib.get_state(pl,local_path,HP.feature_points,HP.distance_between_points)
           
            while not stop:#while not stoped, the loop break if reached the end or the deviation is to big
                #make action:
                Q = net.get_Q([state])#from this state
                Pi = net.get_Pi(state)
                print("velocity1: ",state[0],"Q: ",Q,"PI: ",Pi)#"velocity2: ",state[1],
                #a,one_hot_a = pLib.choose_action(action_space,Pi,steps)#choose action 
                a = pLib.choose_action(action_space,Q[0],steps)
                pl.delta_velocity_command(action_space[a])#update velocity (and steering) and send to simulator. index - index on global path (pl.desired_path)        
                while (not step_now(last_time,HP.step_time)) and not stop: #wait for the next step (after step_time)
                    time.sleep(0.00001)
                    #print(time.time())
                    #tmp = 0
                #get next state:
                pl.simulator.get_vehicle_data()#read data after time step from last action
                local_path = pl.get_local_path(send_path = False,num_of_points = HP.visualized_points)#num_of_points = visualized_points
                next_state = pLib.get_state(pl,local_path,HP.feature_points,HP.distance_between_points)
                #get reward:
                reward = pLib.get_reward(local_path.velocity_limit[0],pl.simulator.vehicle.velocity)
                reward_vec.append(reward)
                Replay.add((state,a,reward,next_state))
              
                rand_state, rand_a, rand_reward, rand_next_state = Replay.sample(HP.batch_size)
                #rand_Q = net.get_Q(rand_state)
                #rand_Q = net.get_targetQ(rand_state)
                rand_next_Q = net.get_Q(rand_next_state)
                rand_next_a = np.argmax(rand_next_Q,axis = 1)#best action from next state according to Q network
                rand_next_targetQ = net.get_targetQ(rand_next_state)
                #rand_Q_ = np.copy(rand_Q)
                rand_targetQ = []
                for i in range(len(rand_state)):
                    #rand_targetQ.append(rand_reward[i] + HP.gamma*np.max(rand_next_Q[i]))
                    rand_targetQ.append(rand_reward[i] + HP.gamma*rand_next_targetQ[i][rand_next_a[i]])

                #update nets:
                
                net.Update_Q(rand_state,rand_a,rand_targetQ)
                print("Q:",net.get_Q([state]))
                net.update_target()
                print("targetQ:",net.get_targetQ([state]))
                #Q_[batch_index] = np.copy(Q)#save Q from last state
                ## print("Q - before: ",Q)
                ##actor-critic:
                #next_Q = net.get_Q(next_state)#from this state
                ##print("correcting with action: ",a)
                #Q_[batch_index][a] = reward +HP.gamma*np.max(next_Q)#compute Q_: reward,a - from last to this state, Q - from this state
                #net.Update_Q([state],Q_)

                #Q_last = net.get_Q(last_state[0])#Q for last state for update policy on last state values
                #Q_loss = net.get_Q_loss(state,Q_[batch_index])
                #print("Q_loss: ",Q_loss[0][a])
                #A = Q[a] - sum(Q)/len(Q)
                #net.Update_policy([state],[one_hot_a],[[A]])

              
          
                
                    
                #TD(lambda):
                ##if(len(state_vec) > HP.num_of_TD_steps):
                ##    future_reward = 0
                ##    for k in range(HP.num_of_TD_steps):
                ##        future_reward += reward_vec[k]*HP.gamma**k 
                ##    net.Update_policy([state_vec[-HP.num_of_TD_steps]],[action_vec[-HP.num_of_TD_steps]],[[future_reward]])
                ##    reward_vec.pop(0)#remove first
                ##    value_vec[-HP.num_of_TD_steps] = [future_reward]#the reward is on the last state and action
                ##reward_vec.append(reward)

                ##state_vec.append(last_state[0])#for one batch only
                ##action_vec.append(one_hot_a)
                value_vec.append([reward])#the reward is on the last state and action

                dataManager.update_real_path(pl = pl,velocity_limit = local_path.velocity_limit[0])#state[0]
                #dataManager.save_additional_data(pl,features = state,action = a)
                state = next_state

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

            if (i % HP.reset_every == 0 and i > 0) or mode == 'kipp': 
                #pl.stop_vehicle()
                pl.simulator.reset_position()
                pl.stop_vehicle()
                #net.save_model()
            if (i % HP.save_every == 0 and i > 0): 
                net.save_model(HP.save_name)
            #reward_sum = 0
            #for item in value_vec: reward_sum+=item[0]
            #if len(value_vec) > 0:
            #    print("mean reward: ",reward_sum/len(value_vec))
            dataManager.comp_rewards(path_num,reward_vec,HP.gamma)
            dataManager.print_data()
            if HP.plot_flag:
                plot.close()
                #plot.plot_path_with_features(dataManager,distance_between_points)
                plot.plot_path(dataManager.real_path)
            pl.restart()#stop vehicle, and initalize real path
            dataManager.restart()
           
            #comp_Pi(net)
            #update policy at the end of the episode:

            #for i in range(len(value_vec)-2, -1, -1):
            #    value_vec[i] += HP.gamma * value_vec[i+1]
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


        #with open(HP.run_data_file_name, 'w') as f:
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
            
    net.save_model(HP.save_name)#model4.ckpt - LINEAR, LINE. model5.ckpt - net, line - good, model6.ckpt - 3 points model7.ckpt -very good
    
