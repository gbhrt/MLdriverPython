
import time
import numpy as np
import library as lib
import classes
import copy
import random
from plot import Plot
import predict_lib
#import subprocess
import math
import os


def comp_MB_acc(net,env,state,acc,steer):
    roll_flag,dev_flag = 0, False
    n = 10
    print("___________________new acc compution____________________________")
    predicted_values,roll_flag,dev_flag = predict_lib.predict_n_next(n,net,env,state,acc,steer,1.0)#try 1.0
    if roll_flag != 0 or dev_flag:#if not ok - try 0.0                                                   
        predicted_values,roll_flag,dev_flag = predict_lib.predict_n_next(n,net,env,state,acc,steer,0.0)
        if roll_flag != 0 or dev_flag:#if not ok - try -1.0   
            predicted_values,roll_flag,dev_flag = predict_lib.predict_n_next(n,net,env,state,acc,steer,-1.0,max_plan_roll = env.max_plan_roll*1.3)#,max_plan_roll = env.max_plan_roll*1.3,max_plan_deviation = 10)
            if roll_flag != 0 or dev_flag:
                next_acc = -1.0
            else:#-1.0 is ok
                next_acc = -1.0
        else:# 0.0 is ok
            next_acc = 0.0
    else:#1.0 is ok
        next_acc = 1.0
    #print("________________________end________________________________")
    emergency_pred_vec = predicted_values
    return next_acc,predicted_values,emergency_pred_vec,roll_flag,dev_flag

def train(env,HP,Agent,dataManager,guiShared,seed = None): 
    

    #"""
    #cd C:\Users\gavri\Desktop\sim_15_3_18
    #sim15_3_18 -quit -batchmode -nographics
    #"""

    #subprocess.Popen('C:\\Users\\gavri\\Desktop\\sim_15_3_18\\sim15_3_18 -quit -batchmode -nographics')
  
    #pre-defined parameters:
    Agent.start_training()
    if seed != None:
        HP.seed = seed
    ###################
    total_step_count = 0
    random_action_flag = True
    #env = environment1.OptimalVelocityPlanner(HP)
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
    for i in range(Agent.trainHP.num_of_runs): #number of runs - run ends at the end of the main path and if vehicle deviation error is to big
        if waitFor.stop == [True] or guiShared.request_exit:
            break
        
        # initialize every episode:
        #if HP.run_random_num != 'inf':
        #    if i > HP.run_random_num:
        #        random_action_flag = False
        step_count = 0
        reward_vec = []
        t1 = 0

        env_state = env.reset(seed = seed)
        if env_state == "error":
            print("error at env.reset")
            i-=1
            continue
        state = Agent.get_state(env_state)

        guiShared.restart()
        if env.error:
            i-=1
            continue
        #episode_start_time = time.time()
        acc,steer = 0.0,0.0
        acc,steer,planningData= Agent.comp_action(state,acc,steer)#for the first time (required because the first time is longer)
        #acc,steer,planningData,roll_flag,dev_flag = Agent.comp_action(env_state,acc,steer,env)#for the first time (required because the first time is longer)

        env.pl.init_timer()
        while  waitFor.stop != [True] and guiShared.request_exit == False:#while not stoped, the loop break if reached the end or the deviation is to big          
            step_count+=1
            #choose and make action:
            if HP.analytic_action:
                acc = float(env.get_analytic_action()[0])#+noise
                steer = env.comp_steer()
                if env.stop_flag:
                    acc = -1.0
                else:
                    last_ind = env.pl.main_index
                    if len(dataManager.real_path.time)>0:
                        last_tim = dataManager.real_path.time[-1]
                    else:
                        last_tim = 0
                env.command(acc,steer)

                with guiShared.Lock:
                    #guiShared.planningData.append(planningData)
                    guiShared.roll = copy.copy(dataManager.roll)
                    guiShared.real_path = copy.deepcopy(dataManager.real_path)
                    guiShared.update_data_flag = True

                next_state, reward, done, info = env.step(acc)#input the estimated next actions to execute after delta t and getting next state
                
            else:#not HP.analytic_action
                acc,steer,planningData = Agent.comp_action(state,acc,steer)#env is temp  ,roll_flag,dev_flag 
                
                #acc,steer,planningData,roll_flag,dev_flag = Agent.comp_action(state,acc,steer)


                #trainShared.algorithmIsIn.clear()#indicates that are ready to take the lock
                #with trainShared.Lock:
                #    trainShared.algorithmIsIn.set()
                #    #net and Replay are shared
                #    #print("time after Lock:",time.clock()-env.lt)
                    
                #    #print("vel_n:",env.pl.simulator.vehicle.wheels[0].vel_n)
                #    #print("before comp_MB_acc time:",time.clock() - env.lt)
                #    #next_acc,next_steer,pred_vec,emergency_pred_vec,roll_flag,dev_flag,emergency_action = comp_MB_action(net,env,state,acc,steer)
                #    next_acc,next_steer,planningData,roll_flag,dev_flag = comp_MB_action(net,env,state,acc,steer)
           
                if env.stop_flag:
                    acc = -1.0
                else:
                    last_ind = env.pl.main_index
                    if len(dataManager.real_path.time)>0:
                        last_tim = dataManager.real_path.time[-1]
                    else:
                        last_tim = 0
                               
                #if dev_flag:
                #    fail = True #save in the Replay buffer that this episode failed
                #    done = True #break

                #t= time.clock()
                with guiShared.Lock:
                    guiShared.planningData.append(planningData)
                    guiShared.roll = copy.copy(dataManager.roll)
                    guiShared.real_path = copy.deepcopy(dataManager.real_path)
                    guiShared.steer = steer
                    guiShared.update_data_flag = True
                #print("update gui time:",time.clock() - t)
                
                env_state, reward, done, info = env.step(acc,steer = steer)#input the estimated next actions to execute after delta t and getting next state
                
                #print("after step 2 time:",time.clock() - env.lt)
                state = Agent.get_state(env_state)

            reward_vec.append(reward)
            # print("after append:", time.time() - env.lt)
            #add data to replay buffer:
            if info[0] == 'kipp' or info[0] == 'deviate':
                fail = True
            else:
                fail = False
            time_step_error = info[1]

            if abs(env.pl.simulator.vehicle.input_time - last_time_stamp-env.step_time) <0.01 and not time_step_error:
                time_error = False
            else:
                time_error = True

            Agent.add_to_replay(state,acc,steer,done,time_error,time_step_error)# fail
            #print("replay:",Agent.Replay.memory[-1])
            last_time_stamp = env.pl.simulator.vehicle.input_time
            
            #state = copy.deepcopy(next_state)
            
            #print("copy state time:",time.clock() - env.lt)
            #if not HP.analytic_action:
            #    acc,steer = copy.copy(next_acc), copy.copy(next_steer)
            if done:
                break

            #end if time
        #end while

        #after episode end:
        env.stop_vehicle_complete()
        total_reward = sum(reward_vec)
        if not HP.gym_flag: #and not HP.noise_flag:
            dataManager.episode_end_mode.append(info[0])
            dataManager.rewards.append(total_reward)
            dataManager.lenght.append(step_count)
            dataManager.add_run_num(i)
            dataManager.add_train_num(global_train_count)
            dataManager.path_seed.append(env.path_seed)#current used seed (for paths)
            dataManager.update_paths()
            relative_reward = dataManager.comp_relative_reward1(env.pl.in_vehicle_reference_path,last_ind,last_tim)
            dataManager.relative_reward.append(relative_reward)
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
            
            #HP.noise_flag =True
        print("episode: ", i, " total reward: ", total_reward, "episode steps: ",step_count)
        
        if not HP.run_same_path:
            seed = int.from_bytes(os.urandom(8), byteorder="big")
        else:#not needed 
            seed = HP.seed

        #if (i % HP.zero_noise_every == 0 and i > 0) or HP.always_no_noise_flag:
        #    HP.noise_flag = False
        #    if HP.test_same_path:
        #        seed = HP.seed

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
    Agent.stop_training()
    Agent.save()
    dataManager.save_data()
        
    
    #del env
    #del HP
    #del net
    #del Replay
    #del actionNoise

    return 
           
       