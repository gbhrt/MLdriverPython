
import time
import numpy as np
import library as lib
from classes import Path
import copy
import random

from plot import Plot
import agent_lib as pLib
#import subprocess
import math
import os


def train(env,HP,net,dataManager,seed = None):
    

    #"""
    #cd C:\Users\gavri\Desktop\sim_15_3_18
    #sim15_3_18 -quit -batchmode -nographics
    #"""

    #subprocess.Popen('C:\\Users\\gavri\\Desktop\\sim_15_3_18\\sim15_3_18 -quit -batchmode -nographics')
  
    #pre-defined parameters:
    if seed != None:
        HP.seed[0] = seed
    ###################
    total_step_count = 0
    evaluation_flag = False
    #env = enviroment1.OptimalVelocityPlanner(HP)
    #env  = gym.make("HalfCheetahBulletEnv-v0")
    np.random.seed(HP.seed[0])
    random.seed(HP.seed[0])
    env.seed(HP.seed[0])
    steps = [0,0]#for random action selection of random number of steps
    waitFor = lib.waitFor()#wait for "enter" in another thread - then stop = true
    if HP.render_flag:
        env.render()
    #env.reset()

   
    #if not HP.skip_run:
        
    #plot = Plot()
    #dataManager = data_manager.DataManager(total_data_names = ['total_reward'],  file = HP.save_name+".txt")
    Replay_fails = pLib.Replay(HP.replay_memory_size)
    if HP.restore_flag:
        Replay_fails.restore(HP.restore_file_path,name = "replay_fails")
    Replay = pLib.Replay(HP.replay_memory_size)
    if HP.restore_flag:
        Replay.restore(HP.restore_file_path)
        

    actionNoise = pLib.OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.action_space.shape[0]),dt = 0.2)#env.step_time
    #if not HP.random_paths_flag:
    #    if pl.load_path(HP.path_name,HP.random_paths_flag) == -1:
    #        stop = [1]
      
    test_path_ind = 0
    global_train_count = 0
    seed = HP.seed[0]
    reduce_vel = 0.0
    for i in range(HP.num_of_runs): #number of runs - run end at the end of the main path and if vehicle deviation error is to big
        if waitFor.stop == [True]:
            break
        # initialize every episode:
        #reduce_vel+=0.01
        #print("reduce_vel: ",reduce_vel)
        step_count = 0
        reward_vec = []
        last_time = [0]
        
        #########################
        #if waitFor.command == [b'3']:
        #    print("repeat last path")
        #    #state = env.reset(path_num = len(dataManager.path_seed) - 1)
        #    state = env.reset(path_num = 1)
        #state = env.reset(path_num = 1234)####################################################################

        state = env.reset(seed = seed)   
        if i == 0:
            print("update nets first time")
            pLib.DDPG([state], [[0]], [0], [state],[False],net,HP)
        #episode_start_time = time.time()
        while  waitFor.stop != [True]:#while not stoped, the loop break if reached the end or the deviation is to big
            step_count+=1
               
            #choose and make action:
            #Q = net.get_Q([state])
            #Pi = net.get_Pi([state])
            #print("velocity1: ",state[0])#,"Q: ",Q)#,"PI: ",Pi)#"velocity2: ",state[1],
            if HP.add_feature_to_action:
                if env.analytic_feature_flag:
                    path_state = state[1:]
                else:
                    path_state = state[:]
                analytic_action = env.comp_analytic_acceleration(path_state)
                noise_range = env.action_space.high[0] - abs(analytic_action)
                vel = path_state[0]
            else:
                noise_range = env.action_space.high[0]
                vel = state[0]
            
            a = net.get_actions(np.reshape(state, (1, env.observation_space.shape[0])))#[[action]] batch, action list
            print("action:",a)#,"analytic_action:",analytic_action)
            noise = actionNoise() * noise_range
            #noise = random.uniform(-1, 1)* noise_range
            

            a = a[0]
            if not evaluation_flag:# HP.noise_flag:
                if vel <0.02:
                    noise = abs(noise)
                a +=  noise#np vectors##########################################################
                dataManager.noise.append(noise)
                print("noise:",noise)
            a = list(np.clip(a,-env.action_space.high[0],env.action_space.high[0]))  
            
            a = [float(a[k]) for k in range(len(a))]   
            #a = [1.0]
            
            #a = [state[0]]# 
            #if HP.noise_flag:


            if HP.add_feature_to_action:
                a[0] += analytic_action

            if HP.analytic_action:# and HP.noise_flag:
                state[0] = state[0]*(1.0- HP.reduce_vel)
                a = [env.comp_analytic_acceleration(state)]#env.analytic_feature_flag must be false
                #print("acc:",a)
           # print("state:", state)
            #a = env.get_analytic_action()
            #print("action: ", a)#,"noise: ",noise)
           
            if not HP.gym_flag:
                env.command(a)
            print("time from get state to execute action:",time.time() - env.lt)

            dataManager.acc_target.append(net.get_target_actions(np.reshape(state, (1, env.observation_space.shape[0])))[0])
            #print("a:",a)
            dataManager.acc.append(a[0])
            Qa = net.get_Qa(np.reshape(state, (1, env.observation_space.shape[0])),[a])[0][0]
            Q0 = net.get_Qa(np.reshape(state, (1, env.observation_space.shape[0])),[[0]])[0][0]
            Q1 = net.get_Qa(np.reshape(state, (1, env.observation_space.shape[0])),[[1.0]])[0][0]
            Qneg1 = net.get_Qa(np.reshape(state, (1, env.observation_space.shape[0])),[[-1.0]])[0][0]

            Qa_target = net.get_targetQa(np.reshape(state, (1, env.observation_space.shape[0])),[a])[0][0]
            Q0_target = net.get_targetQa(np.reshape(state, (1, env.observation_space.shape[0])),[[0]])[0][0]
            Q1_target = net.get_targetQa(np.reshape(state, (1, env.observation_space.shape[0])),[[1.0]])[0][0]
            Qneg1_target = net.get_targetQa(np.reshape(state, (1, env.observation_space.shape[0])),[[-1.0]])[0][0]

            #print("Qa:",Qa,"Q0:",Q0,"Q1",Q1,"Qneg1",Qneg1)
            dataManager.Qa.append(Qa)
            dataManager.Q0.append(Q0)
            dataManager.Q1.append(Q1)
            dataManager.Qneg1.append(Qneg1)

            dataManager.Qa_target.append(Qa_target)
            dataManager.Q0_target.append(Q0_target)
            dataManager.Q1_target.append(Q1_target)
            dataManager.Qneg1_target.append(Qneg1_target)

            if len(Replay.memory) > HP.batch_size and HP.train_flag and not evaluation_flag:############
                if not HP.gym_flag:
                    start_time = time.time()
                    t = start_time
                    last_time = start_time
                    train_count = 0
                    #for _ in range(HP.train_num):
                    while (t - start_time) < env.step_time - (t - last_time)-0.05 and train_count < HP.train_num:  
                        #print(t - start_time, t - last_time)
                        last_time = t
                        train_count += 1
                        #sample from replay buffer:
                        if len(Replay_fails.memory)>0 and HP.sample_ratio != 1.0:
                            rand_state1, rand_a1, rand_reward1, rand_next_state1, rand_end1 = Replay_fails.sample(int(HP.batch_size*(1-HP.sample_ratio)))
                            print("number of done samples:",len(rand_state1))
                            rand_state, rand_a, rand_reward, rand_next_state, rand_end = Replay.sample(HP.batch_size - len(rand_state1))#HP.batch_size*HP.sample_ratio)
                            rand_state+=rand_state1;rand_a+=rand_a1;rand_reward+=rand_reward1;rand_next_state+=rand_next_state1;rand_end+=rand_end1

                            #np.concatenate(rand_state,rand_state1)
                            #np.concatenate(rand_a,rand_a1)
                            #np.concatenate(rand_reward,rand_reward1)
                            #np.concatenate(rand_next_state,rand_next_state1)
                            #np.concatenate(rand_end,rand_end1)
                        else:
                            rand_state, rand_a, rand_reward, rand_next_state, rand_end = Replay.sample(HP.batch_size)#


                        #update neural networs:
                        #pLib.DDQN(rand_state, rand_a, rand_reward, rand_next_state,net,HP)
                        #if HP.add_feature_to_action:
                        #    pLib.DDPG(rand_state, rand_a, rand_reward, rand_next_state,rand_end,net,HP,comp_analytic_acceleration = env.comp_analytic_acceleration)
                        #else:
                        pLib.DDPG(rand_state, rand_a, rand_reward, rand_next_state,rand_end,net,HP)
                        t = time.time()
                        

                    print ("train_count: ", train_count)
                    global_train_count+=train_count
                    
                else:
                    #sample from replay buffer:
                    rand_state, rand_a, rand_reward, rand_next_state, rand_end = Replay.sample(HP.batch_size)
                    #update neural networs:
                    #pLib.DDQN(rand_state, rand_a, rand_reward, rand_next_state,net,HP)
                    pLib.DDPG(rand_state, rand_a, rand_reward, rand_next_state,rand_end,net,HP)


                    
                

            next_state, reward, done, info = env.step(a)
            reward_vec.append(reward)
            #add data to replay buffer:
            #if info == 'kipp':
            #    fail = True
            #else:
            #    fail = False
            #Replay.add((state,a,reward,next_state,fail))#done
            if HP.add_feature_to_action:
                a[0] -= analytic_action
            if not evaluation_flag:
                if done == True and HP.sample_ratio != 1.0:
                    Replay_fails.add((state,a,reward,next_state,done))# 
                    
                else:
                    Replay.add((state,a,reward,next_state,done))#  
                    
            state = next_state

            if done:
                break
                
            #end if time
        #end while

        #after episode end:
        total_reward = 0
        #for k,r in enumerate(reward_vec):
        #    total_reward+=r*HP.gamma**k
        total_reward = sum(reward_vec)
        if not HP.gym_flag and evaluation_flag:#not HP.noise_flag

            #dist = sum(dataManager.real_path.velocity)*env.step_time#integral on velocities
            #analytic_dist = sum(dataManager.real_path.analytic_velocity)*env.step_time
            
            #if len(dataManager.real_path.velocity) > 0:
            #    relative_dist = (dist - analytic_dist)/len(dataManager.real_path.velocity)
            #else:
            #    relative_dist = 0.0
            #dataManager.relative_reward.append(relative_dist)
            dataManager.episode_end_mode.append(info[0])
            dataManager.rewards.append(total_reward)
            dataManager.lenght.append(step_count)
            dataManager.add_run_num(i)
            dataManager.add_train_num(global_train_count)
            dataManager.path_seed.append(env.path_seed)#current used seed (for paths)
            dataManager.update_relative_rewards_and_paths()
            evaluation_flag = False
           # HP.noise_flag =True
        #print("episode time:",time.time()-episode_start_time)
        print("episode: ", i, " total reward: ", total_reward, "episode steps: ",step_count,"file name:",HP.save_file_path)
        
        if not HP.run_same_path:
            seed = int.from_bytes(os.urandom(8), byteorder="big")
        else:#not needed 
            seed = HP.seed[0]

        if (i % HP.evaluation_every == 0 and i > 0) or test_path_ind != 0 or HP.always_no_noise_flag:
            #HP.noise_flag = False
            evaluation_flag = True
            
            if HP.test_same_path:
                seed = HP.seed[test_path_ind]
                print("seed:",seed)
                test_path_ind +=1
                if test_path_ind >= len(HP.seed):
                    test_path_ind = 0

        if (i % HP.save_every == 0 and i > 0): 
            net.save_model(HP.save_file_path)
            Replay.save(HP.save_file_path)
            Replay_fails.save(HP.save_file_path,name = "replay_fails")
            dataManager.save_data()
        if HP.plot_flag and waitFor.command == [b'1']:
            dataManager.plot_all()
            #dataManager.plot.plot('total_reward')#,'curvature'
            #dataManager.plot.plot('curvature')
            #dataManager.plot.plot_path_with_features(dataManager,env.distance_between_points,block = True)
            #dataManager.plot.plot_path(dataManager.real_path,block = True)
        #dataManager.save_readeable_data()
        dataManager.restart()
        #try:
        #    dataManager.comp_rewards(path_num-1,HP.gamma)
        #    dataManager.print_data()
        #except:
        #    print("cannot print data")
        #if HP.plot_flag and command == [b'1']:
        #    plot.close()
        #    plot.plot_path_with_features(dataManager,HP.distance_between_points,block = True)
            #plot.plot_path(dataManager.real_path,block = True)
            
        #dataManager.restart()

    #end all:
    
    env.close()
    net.save_model(HP.save_file_path)
    Replay.save(HP.save_file_path)
    Replay.save(HP.save_file_path,name = "replay_fails")
    dataManager.save_data()
        
    
    #del env
    #del HP
    #del net
    #del Replay
    #del actionNoise

    return 
           
       