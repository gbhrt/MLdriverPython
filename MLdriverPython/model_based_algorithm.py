
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
    roll_flag,dev_flag = False, False
    n = 10
    print("___________________new acc compution____________________________")
    predicted_values,roll_flag,dev_flag = pLib.predict_n_next(n,net,env,state,acc,1.0)#try 1.0
    if roll_flag or dev_flag:#if not ok - try 0.0                                                   
        predicted_values,roll_flag,dev_flag = pLib.predict_n_next(n,net,env,state,acc,0.0)
        if roll_flag or dev_flag:#if not ok - try -1.0   
            predicted_values,roll_flag,dev_flag = pLib.predict_n_next(n,net,env,state,acc,-1.0,max_plan_roll = 0.1,max_plan_deviation = 10)
            if roll_flag or dev_flag:
                next_acc = -1.0
            else:#-1.0 is ok
                next_acc = -1.0
        else:# 0.0 is ok
            next_acc = 0.0
    else:#1.0 is ok
        next_acc = 1.0
    print("________________________end________________________________")
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
    for i in range(HP.num_of_runs): #number of runs - run end at the end of the main path and if vehicle deviation error is to big
        if waitFor.stop == [True] or guiShared.request_exit:
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
        #episode_start_time = time.time()
        steer = 0
        acc = 0.7
        while  waitFor.stop != [True] and guiShared.request_exit == False:#while not stoped, the loop break if reached the end or the deviation is to big          
                step_count+=1
                #choose and make action:
                if HP.analytic_action:
                    acc = float(env.get_analytic_action()[0])#+noise
                    steer = env.comp_steer()
                    env.command(acc,steer)
                    next_state, reward, done, info = env.step(acc)#input the estimated next actions to execute after delta t and getting next state

                else:#not HP.analytic_action
                    trainShared.algorithmIsIn.clear()#indicates that are ready to take the lock
                    with trainShared.Lock:
                        trainShared.algorithmIsIn.set()
                        #net and Replay are shared
                        #print("time after Lock:",time.clock()-env.lt)
                        next_steer = pLib.comp_steer_from_next_state(net,env,state,steer,acc)
                        print("vel_n:",env.pl.simulator.vehicle.wheels[0].vel_n)
                        next_acc,predicted_values,roll_flag,dev_flag = comp_MB_acc(net,env,state,acc)
                        #print("roll_flag:",roll_flag,"dev_flag:",dev_flag)
                        if roll_flag:
                            next_steer = 0.0
                        if dev_flag:
                            fail = True #save in the Replay buffer that this episode failed
                            done = True #break

              
                    with guiShared.Lock:
                        guiShared.predicded_path = [pred[0] for pred in predicted_values]
                        guiShared.state = copy.deepcopy(state)
                        guiShared.steer = steer

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

                    trainShared.algorithmIsIn.clear()#indicates that are ready to take the lock
                    with trainShared.Lock:
                        trainShared.algorithmIsIn.set()
                        Replay.add(copy.deepcopy((state,[acc,steer],next_state,done,fail)))#  

                    next_state['path'] = tmp_next_path

                state = copy.deepcopy(next_state)
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
           
       