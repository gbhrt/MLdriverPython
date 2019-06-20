
import time
import numpy as np
import library as lib
from classes import Path
import copy
import random
import agent_lib as pLib

import math
import os
import classes


def train(env,HP,net_drive,dataManager,net_stabilize = None,guiShared = None,seed = None,global_train_count = 0):
    

    #"""
    #cd C:\Users\gavri\Desktop\sim_15_3_18
    #sim15_3_18 -quit -batchmode -nographics
    #"""

    #subprocess.Popen('C:\\Users\\gavri\\Desktop\\sim_15_3_18\\sim15_3_18 -quit -batchmode -nographics')
  
    #pre-defined parameters:
    if seed != None:
        HP.seed[0] = seed
    ###################

    da = 0.2
    actions = []#for Q evaluation
    for i in np.arange(1,-1-da,-da):
        if HP.env_mode == "SDDPG_pure_persuit":
            actions.append([i])
        else:
            for j in np.arange(1,-1-da,-da):
                actions.append([i,j])
             

    l = int(math.sqrt(len(actions)))

    total_step_count = 0

    evaluation_flag = HP.evaluation_flag

    #env = environment1.OptimalVelocityPlanner(HP)
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
    #Replay_fails = pLib.Replay(HP.replay_memory_size)
    #if HP.restore_flag:
    #    Replay_fails.restore(HP.restore_file_path,name = "replay_fails")
    Replay = pLib.Replay(HP.replay_memory_size)
    if HP.restore_flag and not HP.evaluation_flag:
        Replay.restore(HP.restore_file_path)
        

    actionNoise = pLib.OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.action_space.shape[0]),dt = 0.2)#env.step_time
    #if not HP.random_paths_flag:
    #    if pl.load_path(HP.path_name,HP.random_paths_flag) == -1:
    #        stop = [1]
      
    test_path_ind = 0
    
    seed = HP.seed[0]
    reduce_vel = 0.0
    for i in range(HP.num_of_runs): #number of runs - run end at the end of the main path and if vehicle deviation error is to big
        if waitFor.stop == [True]:
            break
        # initialize every episode:
        dataManager.restart()
        guiShared.restart()
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
        print("seed:",seed)
        if state == 'error':
            print("reset error")
            i = min(0,i-1)
            continue
        if i == 0 and not evaluation_flag:
            print("update nets first time")
            tmp_a = [0] if HP.env_mode == "SDDPG_pure_persuit" else [0,0]
            print("state",state)
            pLib.DDPG([state], [tmp_a], [0], [state],[False],net_drive,HP)
            if HP.stabilize_flag:
                pLib.DDPG([state], [tmp_a], [0], [state],[False],net_stabilize,HP)
        #episode_start_time = time.time()
        while  waitFor.stop != [True]:#while not stoped, the loop break if reached the end or the deviation is to big
            step_count+=1
               
            #choose and make action:
            #Q = net_drive.get_Q([state])
            #Pi = net_drive.get_Pi([state])
            #print("velocity1: ",state[0])#,"Q: ",Q)#,"PI: ",Pi)#"velocity2: ",state[1],
          
            noise_range = env.action_space.high[0]
            emergency_action = False
            if HP.analytic_action:# and HP.noise_flag:
                state[0] = state[0]*(1.0- HP.reduce_vel)
                a = [env.comp_analytic_acceleration(state)]#env.analytic_feature_flag must be false
                #a = env.get_analytic_action()
            else:      
                #Q_matrix = net_drive.get_Qa([state]*len(actions),actions)
                #Q_matrix = Q_matrix.flatten()
                #print(Q)
                #if  HP.env_mode == "SDDPG_pure_persuit":    
                #    Q_matrix = np.reshape(Q_matrix,(len(Q_matrix),1))
                #else:
                    #Q_matrix = np.reshape(Q_matrix,(l,l))

                if HP.DQN_flag:
                    max_Q_ind = np.argmax(Q_matrix)
                    action = actions[max_Q_ind]
                else:
                    action  = net_drive.get_actions(np.reshape(state, (1, env.observation_space.shape[0])))[0]#[[action]] batch, action list
                if HP.stabilize_flag:
                    Q_stabilize = net_stabilize.get_Qa([state],[action])
                    print("Q_stabilize:",Q_stabilize)
                    #Q_matrix_stabilize = net_stabilize.get_Qa([state]*len(actions),actions)
                    #if HP.env_mode != "SDDPG_pure_persuit":
                        #Q_matrix_stabilize = np.reshape(Q_matrix_stabilize.flatten(),(l,l))
                    if Q_stabilize < HP.minQ:
                        emergency_action = True
                        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                        if HP.DQN_flag:
                            action =pLib.get_DQN_action(net_stabilize,state)
                        else:
                            action  = net_stabilize.get_actions(np.reshape(state, (1, env.observation_space.shape[0])))[0]#
                
                
                
                noise = actionNoise() * noise_range

                #if random.random() < HP.epsilon:
                #    noise = random.uniform(-1.0, 1.0)
                #else:
                #     noise = 0;
                #noise = random.uniform(-1, 1)* noise_range
            
                if not evaluation_flag:# HP.noise_flag:
                    if env.pl.simulator.vehicle.velocity[1] <0.1:
                        noise = abs(noise)
                    
                    if not emergency_action:#tmp
                        a = np.array(action) +  noise#np vectors##########################################################
                    else:
                        a = np.array(action)
                    dataManager.noise.append(noise)
                    #print("noise:",noise)
                else:
                    a = action
                a = list(np.clip(a,-env.action_space.high[0],env.action_space.high[0]))  
            
                a = [float(a[k]) for k in range(len(a))]  

                if HP.constant_velocity is not None:
                    #if env.pl.simulator.vehicle.velocity > HP.constant_velocity/env.max_velocity_y: a[0] = 0.0
                    a[0] = 1.0 if env.pl.simulator.vehicle.velocity[1] < HP.constant_velocity/env.max_velocity_y else 0.0
                #a = [1.0]
            
                #a = [state[0]]# 
                #if HP.noise_flag


            #if env.stop_flag:
            #    a[0] = -1#stop after max steps
            #else:
            last_ind = env.pl.main_index
            if len(dataManager.real_path.time)>0:
                last_tim = dataManager.real_path.time[-1]
            else:
                last_tim = 0
                #print("acc:",a)
           # print("state:", state)
            #a = env.get_analytic_action()
            #print("action: ", a)#,"noise: ",noise)
            #print("action:",a)#,"analytic_action:",analytic_action)
            if not HP.gym_flag:
                if HP.env_mode == "SDDPG_pure_persuit":
                    env.command(a[0])
                else:
                    env.command(a[0],steer = a[1])
               # print("time until command: ",time.clock() - env.lt)
            #print("time from get state to execute action:",time.time() - env.lt)

            ##dataManager.acc_target.append(net_drive.get_target_actions(np.reshape(state, (1, env.observation_space.shape[0])))[0])
            ###print("a:",a)
            ##dataManager.acc.append(a[0])
            

            #print(Q)
            #Qa = net_drive.get_Qa(np.reshape(state, (1, env.observation_space.shape[0])),[a])[0][0]
            #Q0 = net_drive.get_Qa(np.reshape(state, (1, env.observation_space.shape[0])),[[0]])[0][0]
            #Q1 = net_drive.get_Qa(np.reshape(state, (1, env.observation_space.shape[0])),[[1.0]])[0][0]
            #Qneg1 = net_drive.get_Qa(np.reshape(state, (1, env.observation_space.shape[0])),[[-1.0]])[0][0]

            ##Qa_target = net_drive.get_targetQa(np.reshape(state, (1, env.observation_space.shape[0])),[a])[0][0]
            ##Q0_target = net_drive.get_targetQa(np.reshape(state, (1, env.observation_space.shape[0])),[[0]])[0][0]
            ##Q1_target = net_drive.get_targetQa(np.reshape(state, (1, env.observation_space.shape[0])),[[1.0]])[0][0]
            ##Qneg1_target = net_drive.get_targetQa(np.reshape(state, (1, env.observation_space.shape[0])),[[-1.0]])[0][0]

            ###print("Qa:",Qa,"Q0:",Q0,"Q1",Q1,"Qneg1",Qneg1)
            ##dataManager.Qa.append(Qa)
            ##dataManager.Q0.append(Q0)
            ##dataManager.Q1.append(Q1)
            ##dataManager.Qneg1.append(Qneg1)

            ##dataManager.Qa_target.append(Qa_target)
            ##dataManager.Q0_target.append(Q0_target)
            ##dataManager.Q1_target.append(Q1_target)
            ##dataManager.Qneg1_target.append(Qneg1_target)

            if len(Replay.memory) > HP.batch_size and HP.train_flag and not evaluation_flag:############
                if not HP.gym_flag:
                    start_time = time.time()
                    t = start_time
                    last_time = start_time
                    train_count = 0
                    #for _ in range(HP.train_num):
                    #while (t - start_time) < env.step_time - (t - last_time)-0.05 and train_count < HP.train_num:  
                    while (t - start_time) < env.step_time - (t - last_time)-0.1 and train_count < HP.train_num:  
                        #print(t - start_time, t - last_time)
                        last_time = t
                        train_count += 1
                        #sample from replay buffer:
                        #if len(Replay_fails.memory)>0 and HP.sample_ratio != 1.0:
                            #rand_state1, rand_a1, rand_reward1, rand_next_state1, rand_end1 = Replay_fails.sample(int(HP.batch_size*(1-HP.sample_ratio)))
                            #print("number of done samples:",len(rand_state1))
                            #rand_state, rand_a, rand_reward, rand_next_state, rand_end = Replay.sample(HP.batch_size - len(rand_state1))#HP.batch_size*HP.sample_ratio)
                            #rand_state+=rand_state1;rand_a+=rand_a1;rand_reward+=rand_reward1;rand_next_state+=rand_next_state1;rand_end+=rand_end1

                        #else:
                        if HP.stabilize_flag:
                            rand_state, rand_a, rand_reward,rand_reward_stabilize, rand_next_state, rand_end = Replay.sample(HP.batch_size)
                        else:
                            rand_state, rand_a, rand_reward, rand_next_state, rand_end = Replay.sample(HP.batch_size)#


                        #update neural networs:
                        if HP.DQN_flag:
                            pLib.DDQN(rand_state, rand_a, rand_reward, rand_next_state,rand_end,net_drive,HP)
                        else:
                            pLib.DDPG(rand_state, rand_a, rand_reward, rand_next_state,rand_end,net_drive,HP)
                        if HP.stabilize_flag:
                            if HP.DQN_flag:
                                pLib.DDQN(rand_state, rand_a, rand_reward_stabilize, rand_next_state,rand_end,net_stabilize,HP)
                            else:
                                pLib.DDPG(rand_state, rand_a, rand_reward_stabilize, rand_next_state,rand_end,net_stabilize,HP)
                        t = time.time()
                        

                        global_train_count+=1
                        if global_train_count % HP.save_every_train_number == 0 and global_train_count > 0:
                            #env.stop_flag = True
                            break
                    if global_train_count % HP.save_every_train_number == 0 and global_train_count > 0:
                            #env.stop_flag = True
                            print("break and save")
                            break
                            #HP.train_flag = False
                #else:
                #    #sample from replay buffer:
                #    rand_state, rand_a, rand_reward, rand_next_state, rand_end = Replay.sample(HP.batch_size)
                #    #update neural networs:
                #    #pLib.DDQN(rand_state, rand_a, rand_reward, rand_next_state,net_drive,HP)
                #    pLib.DDPG(rand_state, rand_a, rand_reward, rand_next_state,rand_end,net_drive,HP)

            if global_train_count > 100000:
                break
                    
                

            if guiShared is not None:
                planningData = classes.planningData()
                planningData.vec_path.append(env.local_path)
                #planningData.vec_Q.append(Q_matrix)# Q_matrix_stabilize
                planningData.action_vec.append(action)
                planningData.action_noise_vec.append(a)
                planningData.vec_emergency_action.append(emergency_action)
                if HP.env_mode == 'DDPG_target':
                    planningData.target_point.append(state[1:5])
                with guiShared.Lock:
                    guiShared.planningData.append(planningData)
                    guiShared.roll = copy.copy(dataManager.roll)
                    guiShared.real_path = copy.deepcopy(dataManager.real_path)
                    #guiShared.action = action
                    guiShared.steer = env.pl.simulator.vehicle.steering
                    guiShared.update_data_flag = True

            if HP.stabilize_flag:
                next_state, reward,reward_stabilize, done, info = env.step(stabilize_flag = True)
            else:
                next_state, reward, done, info = env.step()
            
            reward_vec.append(reward)
            #add data to replay buffer:
            #if info == 'kipp':
            #    fail = True
            #else:
            #    fail = False
            #Replay.add((state,a,reward,next_state,fail))#done

            if not evaluation_flag:
                time_step_error = info[1]
                if not time_step_error:
                    print('reward:',reward)
                    if HP.stabilize_flag:
                        Replay.add((state,a,reward,reward_stabilize,next_state,done))
                    else:
                        Replay.add((state,a,reward,next_state,done))# 
                #if done == True and HP.sample_ratio != 1.0:
                #    Replay_fails.add((state,a,reward,next_state,done))# 
                    
                #else:
                #    Replay.add((state,a,reward,next_state,done))#  
                    
            state = next_state

            if done:
                break
                
            #end if time

        #end while
        #if info[0] == 'kipp':
        #    time.sleep(5)
        env.stop_vehicle_complete()
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
            dataManager.update_paths()
            relative_reward = dataManager.comp_relative_reward1(env.pl.in_vehicle_reference_path,last_ind,step_count*env.step_time)
            print("relative_reward: ", relative_reward)
            dataManager.relative_reward.append(relative_reward)
            evaluation_flag = HP.evaluation_flag
           # HP.noise_flag =True
        #print("episode time:",time.time()-episode_start_time)
            with guiShared.Lock:
                guiShared.episodes_data.append(relative_reward)
                #print("planningData.vec_emergency_action:",planningData.vec_emergency_action,'info[0]:',info[0])
                if info[0] == 'kipp' or info[0] == 'deviate':
                    guiShared.episodes_fails.append(1)
                elif any(guiShared.planningData.vec_emergency_action):
                    guiShared.episodes_fails.append(2)
                else:
                    guiShared.episodes_fails.append(0)
                guiShared.update_episodes_flag = True
        print("episode:", i, "episode steps:",step_count,"file name:",HP.save_file_path)
        
        if not HP.run_same_path:
            seed = int.from_bytes(os.urandom(8), byteorder="big")
        else:#not needed 
            seed = HP.seed[0]

        if (i % HP.evaluation_every == 0 and i > 0) or test_path_ind != 0 or HP.always_no_noise_flag or guiShared.evaluate:
            #HP.noise_flag = False
            evaluation_flag = True
            if HP.test_same_path:
                test_path_ind +=1
                seed = HP.seed[test_path_ind]
                print("seed:",seed)
                if test_path_ind >= len(HP.seed):
                    test_path_ind = 0

        if (global_train_count % HP.save_every_train_number == 0 and global_train_count > 0):
            HP.train_flag = True

            net_drive.save_model(HP.save_file_path,name = 'tf_model_'+str(global_train_count))
            if HP.stabilize_flag:
                net_stabilize.save_model(HP.save_file_path,name = 'tf_model_stabilize_'+str(global_train_count))

        if (i % HP.save_every == 0 and i > 0): 
            if not HP.evaluation_flag:
                net_drive.save_model(HP.save_file_path)
                if HP.stabilize_flag:
                    net_stabilize.save_model(HP.save_file_path,name = 'tf_model_stabilize')
                Replay.save(HP.save_file_path)
            #Replay_fails.save(HP.save_file_path,name = "replay_fails")
                dataManager.save_data()
        if HP.plot_flag and waitFor.command == [b'1']:
            dataManager.plot_all()
 
        

        while guiShared.pause_after_episode_flag:
            time.sleep(0.1)

        

      
            
        #dataManager.restart()

    #end all:
    
    env.close()
    net_drive.save_model(HP.save_file_path)
    if HP.stabilize_flag:
        net_stabilize.save_model(HP.save_file_path,name = 'tf_model_stabilize')
    Replay.save(HP.save_file_path)
    #Replay.save(HP.save_file_path,name = "replay_fails")
    dataManager.save_data()
        
    
    #del env
    #del HP
    #del net_drive
    #del Replay
    #del actionNoise

    return 
           
       