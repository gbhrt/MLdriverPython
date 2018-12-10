
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

def comp_MB_acc(net,env,state,acc):
    fail_flag = False
    n = 10
    print("___________________new acc compution____________________________")
    predicted_values,roll_flag = pLib.predict_n_next(n,net,env,state,acc,1.0)#try 1.0
    if roll_flag:#if not ok - try 0.0                                                   
        predicted_values,roll_flag = pLib.predict_n_next(n,net,env,state,acc,0.0)
        if roll_flag:#if not ok - try -1.0   
            predicted_values,roll_flag = pLib.predict_n_next(n,net,env,state,acc,-1.0,max_plan_roll = 0.1)
            if roll_flag:
                next_acc = -1.0
                fail_flag = True
            else:#-1.0 is ok
                next_acc = -1.0
        else:# 0.0 is ok
            next_acc = 0.0
    else:#1.0 is ok
        next_acc = 1.0
    print("________________________end________________________________")
    return next_acc,predicted_values,fail_flag

def train(env,HP,net,Replay,dataManager,trainShared,guiShared,seed = None):
    

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
        t1 = 0


        state = env.reset(seed = seed)   
        if env.error:
            i-=1
            continue
        if i == 0:
            print("update nets first time")
            pLib.model_based_update([state], [[0,0]], [state],[False],net,HP,env)

        #episode_start_time = time.time()
        steer = 0
        acc = 0.7
        while  waitFor.stop != [True]:#while not stoped, the loop break if reached the end or the deviation is to big
           
                step_count+=1
               
                #choose and make action:


                if HP.analytic_action:
                    acc = float(env.get_analytic_action()[0])#+noise
                    steer = env.comp_steer()
                    env.command(acc,steer)
                    next_state, reward, done, info = env.step(acc)#input the estimated next actions to execute after delta t and getting next state


                else:#not HP.analytic_action
                     #print("time before Lock:",time.clock()-env.lt)
                    #print("try lock1")
                    trainShared.algorithmIsIn.clear()#indicates that are ready to take the lock
                    with trainShared.Lock:
                       # print("locked1")
                        trainShared.algorithmIsIn.set()
                        #net and Replay are shared
                #print("time after Lock:",time.clock()-env.lt)
                        next_steer = pLib.comp_steer_from_next_state(net,env,state,steer,acc)
                        #print("next state steer:",steer)
                
                        fail_flag = False
                        next_acc,predicted_values,fail_flag = comp_MB_acc(net,env,state,acc)
                        print("fail_flag:",fail_flag)
                        if fail_flag:
                            next_steer = 0.0
              
                    with guiShared.Lock:
                        guiShared.predicded_path = [pred[0] for pred in predicted_values]
                        guiShared.state = copy.deepcopy(state)
                        guiShared.steer = steer
                
                

                    #print("time before command:",time.time()-env.lt)

                    #print("before step:", time.time() - t1)
                    next_state, reward, done, info = env.step(next_acc,steer = next_steer)#input the estimated next actions to execute after delta t and getting next state

            
                reward_vec.append(reward)
                # print("after append:", time.time() - env.lt)
                #add data to replay buffer:
                if info[0] == 'kipp':
                    fail = True
                else:
                    fail = False
                time_step_error = info[1]

 
                if not time_step_error:
                    tmp_next_path = next_state['path']
                    #  print("after copy1:", time.time() - t1)
                    state['path'] = []
                    next_state['path'] = []
                    #  print("after copy2:", time.time() - t1)
                    #t = time.time()
                    #print (t - lt)
                    #lt = t
                    #print("try lock2")
                    trainShared.algorithmIsIn.clear()#indicates that are ready to take the lock
                    with trainShared.Lock:
                       # print("locked2")
                        trainShared.algorithmIsIn.set()
                        Replay.add(copy.deepcopy((state,[acc,steer],next_state,done,fail)))#  

                    next_state['path'] = tmp_next_path

                    # print("after add:", time.time() - t1)

                state = copy.deepcopy(next_state)
                if not HP.analytic_action:
                    acc,steer = copy.copy(next_acc), copy.copy(next_steer)
                #print("t9:", time.time() - t1)
                if done:
                    break

            
           # print("end loop:", time.time() - t1)
            #end if time
        #end while

        #after episode end:
        #total_reward = 0
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
            dataManager.episode_end_mode.append(info[0])
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
            #net.save_model(HP.save_file_path)
            #Replay.save(HP.save_file_path)
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
    trainShared.train = False
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
           
       