
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

def comp_MB_acc(net,env,state,acc,steer):
    roll_flag,dev_flag = 0, False
    n = 10
    print("___________________new acc compution____________________________")
    predicted_values,roll_flag,dev_flag = pLib.predict_n_next(n,net,env,state,acc,steer,1.0)#try 1.0
    if roll_flag != 0 or dev_flag:#if not ok - try 0.0                                                   
        predicted_values,roll_flag,dev_flag = pLib.predict_n_next(n,net,env,state,acc,steer,0.0)
        if roll_flag != 0 or dev_flag:#if not ok - try -1.0   
            predicted_values,roll_flag,dev_flag = pLib.predict_n_next(n,net,env,state,acc,steer,-1.0,max_plan_roll = env.max_plan_roll*1.3)#,max_plan_roll = env.max_plan_roll*1.3,max_plan_deviation = 10)
            if roll_flag != 0 or dev_flag:
                next_acc = -1.0
            else:#-1.0 is ok
                next_acc = -1.0
        else:# 0.0 is ok
            next_acc = 0.0
    else:#1.0 is ok
        next_acc = 1.0
    #print("________________________end________________________________")
    return next_acc,predicted_values,roll_flag,dev_flag

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
    last_time_stamp = 0
    for i in range(HP.num_of_runs): #number of runs - run ends at the end of the main path and if vehicle deviation error is to big
        if waitFor.stop == [True] or guiShared.request_exit:
            break
        
        # initialize every episode:
        #if HP.run_random_num != 'inf':
        #    if i > HP.run_random_num:
        #        random_action_flag = False
        step_count = 0
        reward_vec = []
        t1 = 0

        state = env.reset(seed = seed)   
        if env.error:
            i-=1
            continue
        #episode_start_time = time.time()
        steer = 0
        acc = 1.0
        while  waitFor.stop != [True] and guiShared.request_exit == False:#while not stoped, the loop break if reached the end or the deviation is to big          
            step_count+=1
            #choose and make action:
            if HP.analytic_action:
                acc = float(env.get_analytic_action()[0])#+noise
                steer = env.comp_steer()
                if env.stop_flag:
                    acc = -1
                env.command(acc,steer)
                next_state, reward, done, info = env.step(acc)#input the estimated next actions to execute after delta t and getting next state
                
            else:#not HP.analytic_action
                trainShared.algorithmIsIn.clear()#indicates that are ready to take the lock
                with trainShared.Lock:
                    trainShared.algorithmIsIn.set()
                    #net and Replay are shared
                    #print("time after Lock:",time.clock()-env.lt)
                    next_steer = pLib.comp_steer_from_next_state(net,env,state,steer,acc)
                    #print("vel_n:",env.pl.simulator.vehicle.wheels[0].vel_n)
                    #print("before comp_MB_acc time:",time.clock() - env.lt)
                    next_acc,predicted_values,roll_flag,dev_flag = comp_MB_acc(net,env,state,acc,steer)
                    dataManager.planed_roll = np.array(predicted_values)[:,3]
                    dataManager.planned_roll_var = np.array(predicted_values)[:,4]
                    print("next_acc:",next_acc)
                    #print("after comp_MB_acc time:",time.clock() - env.lt)
                    #print("roll_flag:",roll_flag,"dev_flag:",dev_flag)
                    if env.stop_flag:
                        next_acc = -1
                    if roll_flag != 0:
                        next_steer = 0 # math.copysign(0.7,roll_flag)
                 
                        print("emergency steering")
                    if dev_flag:
                        fail = True #save in the Replay buffer that this episode failed
                        done = True #break

                #t= time.clock()
                with guiShared.Lock:
                    guiShared.predicded_path = [pred[0] for pred in predicted_values]
                    guiShared.state = copy.deepcopy(state)
                    guiShared.steer = steer
                #print("update gui time:",time.clock() - t)
                
                next_state, reward, done, info = env.step(next_acc,steer = next_steer)#input the estimated next actions to execute after delta t and getting next state
                #print("after step 2 time:",time.clock() - env.lt)
                
            reward_vec.append(reward)
            # print("after append:", time.time() - env.lt)
            #add data to replay buffer:
            if info[0] == 'kipp' or info[0] == 'deviate':
                fail = True
            else:
                fail = False
            time_step_error = info[1]

            if abs(env.pl.simulator.vehicle.input_time - last_time_stamp-env.step_time) <0.01 and not time_step_error:
            #if not time_step_error:#save state to the replay buffer (without the path)            
                #tmp_next_path = next_state['path']
                ##  print("after copy1:", time.time() - t1)
                #state['path'] = []
                #next_state['path'] = []

                trainShared.algorithmIsIn.clear()#indicates that are ready to take the lock
                with trainShared.Lock:
                    trainShared.algorithmIsIn.set()
                    
                    X,Y_ =env.create_XY_([state],[[acc,steer]],[next_state])#normalized data
                    dict_X = env.X_to_X_dict(X[0])
                    dict_Y_ = env.Y_to_Y_dict(Y_[0])
                    for name in env.copy_Y_to_X_names:
                        dict_Y_[name] -= dict_X[name]
                    X = env.dict_X_to_X(dict_X)
                    Y_ = env.dict_Y_to_Y(dict_Y_)
                    Replay.add(copy.deepcopy((X,Y_,done,fail)))# 
                    #Replay.add(copy.deepcopy((state,[acc,steer],next_state,done,fail)))#  

                #next_state['path'] = tmp_next_path
                #print("add to replay time:",time.clock() - env.lt)
            else:
                print("not saving to replay buffer")
            last_time_stamp = env.pl.simulator.vehicle.input_time
            
            state = copy.deepcopy(next_state)
            #print("copy state time:",time.clock() - env.lt)
            if not HP.analytic_action:
                acc,steer = copy.copy(next_acc), copy.copy(next_steer)
            if done:
                break

            #end if time
        #end while

        #after episode end:
        total_reward = sum(reward_vec)
        if not HP.gym_flag and not HP.noise_flag:
            dataManager.episode_end_mode.append(info[0])
            dataManager.rewards.append(total_reward)
            dataManager.lenght.append(step_count)
            dataManager.add_run_num(i)
            dataManager.add_train_num(global_train_count)
            dataManager.path_seed.append(env.path_seed)#current used seed (for paths)
            dataManager.update_relative_rewards_and_paths()
            
            HP.noise_flag =True
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
            dataManager.save_data()
        if HP.plot_flag and waitFor.command == [b'1']:
            dataManager.plot_all()

        while guiShared.pause_after_episode_flag:
            time.sleep(0.1)
        #dataManager.save_readeable_data()
        dataManager.restart()

     

    #end all:
    
    env.close()
    trainShared.train = False
    net.save_model(HP.save_file_path)
    Replay.save(HP.save_file_path)
    dataManager.save_data()
        
    
    #del env
    #del HP
    #del net
    #del Replay
    #del actionNoise

    return 
           
       