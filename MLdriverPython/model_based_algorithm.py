
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


def train(env,HP,net,dataManager,shared,seed = None):
    

    #"""
    #cd C:\Users\gavri\Desktop\sim_15_3_18
    #sim15_3_18 -quit -batchmode -nographics
    #"""

    #subprocess.Popen('C:\\Users\\gavri\\Desktop\\sim_15_3_18\\sim15_3_18 -quit -batchmode -nographics')
  
    #pre-defined parameters:
    if seed != None:
        HP.seed = seed
    ###################
    total_step_count = 0
    random_action_flag = True
    #env = enviroment1.OptimalVelocityPlanner(HP)
    #env  = gym.make("HalfCheetahBulletEnv-v0")
    np.random.seed(HP.seed)
    random.seed(HP.seed)
    env.seed(HP.seed)
    steps = [0,0]#for random action selection of random number of steps
    waitFor = lib.waitFor()#wait for "enter" in another thread - then stop = true
    if HP.render_flag:
        env.render()
    #env.reset()

   
    #if not HP.skip_run:
        
    #plot = Plot()
    #dataManager = data_manager.DataManager(total_data_names = ['total_reward'],  file = HP.save_name+".txt")
    Replay = pLib.Replay(HP.replay_memory_size)
    #Replay1 = pLib.Replay(HP.replay_memory_size)
    if HP.restore_flag:
        Replay.restore(HP.restore_file_path)
        #Replay1.restore(HP.restore_file_path)
        

    actionNoise = pLib.OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.action_space.shape[0]),dt = 0.2)#env.step_time
    #if not HP.random_paths_flag:
    #    if pl.load_path(HP.path_name,HP.random_paths_flag) == -1:
    #        stop = [1]
      

    global_train_count = 0
    seed = HP.seed

    lt = 0#temp
    for i in range(HP.num_of_runs): #number of runs - run end at the end of the main path and if vehicle deviation error is to big
        if waitFor.stop == [True]:
            break
        # initialize every episode:
        if HP.run_random_num != 'inf':
            if i > HP.run_random_num:
                random_action_flag = False
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
            pLib.model_based_update([state], [[0,0]], [state],[False],net,HP)

        #episode_start_time = time.time()
        while  waitFor.stop != [True]:#while not stoped, the loop break if reached the end or the deviation is to big
            step_count+=1
               
            #choose and make action:

            noise_range = env.action_space.high[0]
            noise = actionNoise() * noise_range
            if random_action_flag:
                a = noise
            else:
                a = pLib.comp_model_based_action(state)



            if HP.noise_flag:
                a +=  noise#np vectors##########################################################
                dataManager.noise.append(noise)
                #print("noise:",noise)

            
            #a = [env.comp_analytic_acceleration(state)]#env.analytic_feature_flag must be false
            a = env.get_analytic_action()+noise
            a = list(np.clip(a,-env.action_space.high[0],env.action_space.high[0]))
            a = [float(a[k]) for k in range(len(a))]  
            #a = [env.comp_analytic_acc_compare()]
           # print("state:", state)

            #print("action: ", a)#,"noise: ",noise)
            dataManager.acc.append(a)
            if not HP.gym_flag:
                steer = env.comp_steer()
                n = 10
                max_roll = 0.05
                predicted_values,roll_flag = pLib.predict_n_next(n,net,state,max_roll)
                if roll_flag:
                    a = [-0.7]
                else:
                    a = [0.7]
                shared.predicded_path = [pred[0] for pred in predicted_values]
                steer = lib.comp_steer_general(state['path'],[0,0],0,state['vel'])
                shared.state = state
                steer_command = env.command(a,steer)#steer

            if len(Replay.memory) > HP.batch_size and HP.train_flag:############
                if not HP.gym_flag:
                    start_time = time.time()
                    t = start_time
                    last_time = start_time
                    train_count = 0
                    #for _ in range(HP.train_num):
                    while (t - start_time) < env.step_time - (t - last_time)-0.05 and train_count < HP.train_num:  
                        #print("time:",t - start_time)#, t - last_time)
                        last_time = t
                        train_count += 1
                        #sample from replay buffer:
                        rand_state, rand_a, rand_next_state, rand_end,_ = Replay.sample(HP.batch_size)
                        #update neural networs:
                        #pLib.DDQN(rand_state, rand_a, rand_reward, rand_next_state,net,HP)
                        pLib.model_based_update(rand_state, rand_a, rand_next_state,rand_end,net,HP)
                        
                        t = time.time()
                        

                    #print ("train_count: ", train_count)
                    global_train_count+=train_count 
                    
                else:
                    #sample from replay buffer:
                    rand_state, rand_a,rand_next_state, rand_end,_ = Replay.sample(HP.batch_size)
                    #update neural networs:
                    #pLib.DDQN(rand_state, rand_a, rand_reward, rand_next_state,net,HP)
                    pLib.model_based_update(rand_state, rand_a, rand_next_state,rand_end,net,HP)


                    
            

            next_state, reward, done, info = env.step(a)

            reward_vec.append(reward)
            #add data to replay buffer:
            if info == 'kipp':
                fail = True
            else:
                fail = False
            #Replay.add((state,a,reward,next_state,fail))#done
            replay_state,replay_next_state = copy.deepcopy(state),copy.deepcopy(next_state)
            replay_state['path'] = []
            replay_next_state['path'] = []

            #t = time.time()
            #print (t - lt)
            #lt = t
            Replay.add((replay_state,[a[0],steer_command],replay_next_state,done,fail))#  
            #Replay1.add((state['vel'],state['steer'],a[0],steer_command,
            #            next_state['rel_pos'][0],next_state['rel_pos'][1],next_state['rel_ang'],next_state['vel'],next_state['steer']))#  
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
        if not HP.gym_flag and not HP.noise_flag:

            #dist = sum(dataManager.real_path.velocity)*env.step_time#integral on velocities
            #analytic_dist = sum(dataManager.real_path.analytic_velocity)*env.step_time
            
            #if len(dataManager.real_path.velocity) > 0:
            #    relative_dist = (dist - analytic_dist)/len(dataManager.real_path.velocity)
            #else:
            #    relative_dist = 0.0
            #dataManager.relative_reward.append(relative_dist)
            dataManager.episode_end_mode.append(info)
            dataManager.rewards.append(total_reward)
            dataManager.lenght.append(step_count)
            dataManager.add_run_num(i)
            dataManager.add_train_num(global_train_count)
            dataManager.path_seed.append(env.path_seed)#current used seed (for paths)
            dataManager.update_relative_rewards_and_paths()
            
            HP.noise_flag =True
        #print("episode time:",time.time()-episode_start_time)
        print("episode: ", i, " total reward: ", total_reward, "episode steps: ",step_count)
        
        if not HP.run_same_path:
            seed = int.from_bytes(os.urandom(8), byteorder="big")
        else:#not needed 
            seed = HP.seed

        if (i % HP.zero_noise_every == 0 and i > 0) or HP.always_no_noise_flag:
            HP.noise_flag = False
            if HP.test_same_path:
                seed = HP.seed

        if (i % HP.save_every == 0 and i > 0): 
            net.save_model(HP.save_file_path)
            Replay.save(HP.save_file_path)
            #Replay1.save(HP.save_file_path)
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
    #Replay1.save(HP.save_file_path)
    dataManager.save_data()
        
    
    #del env
    #del HP
    #del net
    #del Replay
    #del actionNoise

    return 
           
       