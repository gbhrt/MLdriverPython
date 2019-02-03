
import time
import numpy as np
import library as lib
import classes
import copy
import random

from plot import Plot
import agent_lib as pLib
#import subprocess
import math
import os

def comp_MB_action(net,env,state,acc,steer):
    print("___________________new acc compution____________________________")
    X_dict,abs_pos,abs_ang = pLib.initilize_prediction(env,state,acc,steer)
    pred_vec = [[abs_pos,abs_ang,state['vel_y'],state['roll'],0,steer,acc]]
    delta_var = 0.0#0.002
    max_plan_roll = env.max_plan_roll
    max_plan_deviation = env.max_plan_deviation
    roll_var = delta_var
    #stability at the current state:
    roll_flag,dev_flag = pLib.check_stability(env,state['path'],0,abs_pos,X_dict['roll'],roll_var = roll_var,max_plan_roll = max_plan_roll,max_plan_deviation = max_plan_deviation)
    if roll_flag == 0 and not dev_flag:
        #predict the next unavoidable state (actions already done):
        X_dict,abs_pos,abs_ang = pLib.predict_one_step(net,env,copy.copy(X_dict),abs_pos,abs_ang)
        roll_var +=delta_var
        vel = env.denormalize(X_dict['vel_y'],"vel_y")
        index = lib.find_index_on_path(state['path'],abs_pos)    
        steer = pLib.steer_policy(abs_pos,abs_ang,state['path'],index,vel)
        X_dict["steer_action"] = env.normalize(steer,"steer_action")
        roll = env.denormalize(X_dict["roll"],"roll")
        #pred_vec.append([abs_pos,abs_ang,vel,roll,roll_var,steer,acc])#acc is not updated
        #stability at the next state (unavoidable state):
        roll_flag,dev_flag = pLib.check_stability(env,state['path'],index,abs_pos,X_dict['roll'],roll_var = roll_var,max_plan_roll = max_plan_roll,max_plan_deviation = max_plan_deviation)
        
        if roll_flag == 0 and not dev_flag:#first step was Ok
            #integration on the next n steps:
            n = 10
            acc_to_try =[1.0,0.0,-1.0]
            for i,try_acc in enumerate(acc_to_try):
                X_dict["acc_action"] = try_acc
                if i == len(acc_to_try)-1:
                    max_plan_roll = env.max_plan_roll#*2.0
                print("try acc:",try_acc)

                pred_vec_n,roll_flag,dev_flag = pLib.predict_n_next1(n,net,env,copy.copy(X_dict),abs_pos,abs_ang,state['path'],max_plan_roll = max_plan_roll,roll_var = roll_var,delta_var = delta_var,max_plan_deviation = max_plan_deviation)
                print("roll_flag:",roll_flag,"dev_flag",dev_flag)
                if (roll_flag == 0 and not dev_flag) or i == len(acc_to_try)-1:
                    #check if the emergency policy is save after applying the action on the first step
                    emergency_pred_vec_n,emergency_roll_flag,emergency_dev_flag = pLib.predict_n_next1(n,net,env,copy.copy(X_dict),abs_pos,abs_ang,state['path'],emergency_flag = True,max_plan_roll = max_plan_roll,roll_var = roll_var,delta_var = delta_var,max_plan_deviation = max_plan_deviation)
                    if emergency_roll_flag == 0 and not emergency_dev_flag:#regular policy and emergency policy are ok:
                        next_acc = try_acc
                        next_steer = steer
                        emergency_action = False
                        break
                #emergency_pred_vec_n,emergency_roll_flag,emergency_dev_flag = [],0,False
                #if (roll_flag == 0 and not dev_flag):
                    
                #    next_acc = try_acc
                #    next_steer = steer
                #    break
                #else the tried action wasn't ok, and must try again
            #pred_vec.append([abs_pos,abs_ang,vel,roll,roll_var,next_steer,next_acc])#the second prediction, the first is the initial state
            #emergency_pred_vec = pred_vec+emergency_pred_vec_n
            #pred_vec+=pred_vec_n
            if (emergency_roll_flag != 0 or emergency_dev_flag) and (roll_flag != 0 or dev_flag):#both regular and emergency failed:
                print("no solution! will fail!!!")

            if (emergency_roll_flag != 0 or emergency_dev_flag) and (roll_flag == 0 and not dev_flag):
                print("solution only for the regular policy - emergency fail")
                next_acc = acc_to_try[-1]#can only be the last
                next_steer = steer
                emergency_action = False

        else:# first step will cause fail
            print("unavoidable fail - after first step")
            #emergency_pred_vec = pred_vec#save just initial state and sirst step
            emergency_pred_vec_n = []
            pred_vec_n = []


        #emergency_roll_flag,emergency_dev_flag = True,True#for the case that already failed

        if (roll_flag != 0 or dev_flag):# and (emergency_roll_flag == 0 and not emergency_dev_flag):#if just emergency is OK:
            next_steer = pLib.emergency_steer_policy()# steering from the next state, send it to the vehicle at the next state
            next_acc = pLib.emergency_acc_policy()
            emergency_action = True
            print("emergency policy is executed!")

        pred_vec.append([abs_pos,abs_ang,vel,roll,roll_var,next_steer,next_acc])#the second prediction, the first is the initial state
        emergency_pred_vec = pred_vec+emergency_pred_vec_n
        pred_vec+=pred_vec_n

    else:#initial state is already unstable
        print("already failed - before first step")
        #emergency_pred_vec = pred_vec#save just initial state
        next_steer = pLib.emergency_steer_policy()# steering from the next state, send it to the vehicle at the next state
        next_acc = pLib.emergency_acc_policy()
        emergency_action = True
        emergency_pred_vec = pred_vec

    planningData = classes.planningData()
    np_pred_vec = np.array(pred_vec)
    np_emergency_pred_vec = np.array(emergency_pred_vec)

    planningData.vec_planned_roll.append(np_pred_vec[:,3])
    planningData.vec_planned_roll_var.append(np_pred_vec[:,4])
    planningData.vec_emergency_planned_roll.append(np_emergency_pred_vec[:,3])
    planningData.vec_emergency_planned_roll_var.append(np_emergency_pred_vec[:,4])
    planningData.vec_planned_vel.append(np_pred_vec[:,2])
    planningData.vec_emergency_planned_vel.append(np_emergency_pred_vec[:,2])
    planningData.vec_planned_acc.append(np_pred_vec[:,6])
    planningData.vec_planned_steer.append(np_pred_vec[:,5])
    planningData.vec_emergency_planned_acc.append(np_emergency_pred_vec[:,6])
    planningData.vec_emergency_planned_steer.append(np_emergency_pred_vec[:,5])
    planningData.vec_emergency_action.append(emergency_action)


    planningData.vec_path.append(state['path'])
    planningData.vec_predicded_path.append(np_pred_vec[:,0])
    planningData.vec_emergency_predicded_path.append(np_emergency_pred_vec[:,0])

    #return next_acc,next_steer,pred_vec,emergency_pred_vec,roll_flag,dev_flag,emergency_action
    return next_acc,next_steer,planningData,roll_flag,dev_flag

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
    emergency_pred_vec = predicted_values
    return next_acc,predicted_values,emergency_pred_vec,roll_flag,dev_flag

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
        guiShared.restart()
        if env.error:
            i-=1
            continue
        #episode_start_time = time.time()
        steer = 0
        acc = 1.0
        env.pl.init_timer()
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
                    
                    #print("vel_n:",env.pl.simulator.vehicle.wheels[0].vel_n)
                    #print("before comp_MB_acc time:",time.clock() - env.lt)
                    #next_acc,next_steer,pred_vec,emergency_pred_vec,roll_flag,dev_flag,emergency_action = comp_MB_action(net,env,state,acc,steer)
                    next_acc,next_steer,planningData,roll_flag,dev_flag = comp_MB_action(net,env,state,acc,steer)
                    #next_steer = pLib.comp_steer_from_next_state(net,env,state,steer,acc)
                    #next_acc,pred_vec,emergency_pred_vec,roll_flag,dev_flag = comp_MB_acc(net,env,state,acc,steer)
                    #if roll_flag != 0:
                    #    next_steer = 0 # math.copysign(0.7,roll_flag)
                    #print("emergency steering")
                    #dataManager.vec_planned_roll.append(np.array(pred_vec)[:,3])
                    #dataManager.vec_planned_roll_var.append(np.array(pred_vec)[:,4])

                    #dataManager.vec_emergency_planned_roll.append(np.array(emergency_pred_vec)[:,3])
                    #dataManager.vec_emergency_planned_roll_var.append(np.array(emergency_pred_vec)[:,4])
                    #print("next_acc:",next_acc)
                    #print("after comp_MB_acc time:",time.clock() - env.lt)
                    #print("roll_flag:",roll_flag,"dev_flag:",dev_flag)
                    if env.stop_flag:
                        next_acc = -1
                 
                 
                        
                    if dev_flag:
                        fail = True #save in the Replay buffer that this episode failed
                        done = True #break

                #t= time.clock()
                with guiShared.Lock:
                    #guiShared.planningData.vec_planned_roll.append(copy.copy(np.array(pred_vec)[:,3]))
                    #guiShared.planningData.vec_planned_roll_var.append(copy.copy(np.array(pred_vec)[:,4]))
                    #guiShared.planningData.vec_emergency_planned_roll.append(copy.copy(np.array(emergency_pred_vec)[:,3]))
                    #guiShared.planningData.vec_emergency_planned_roll_var.append(copy.copy(np.array(emergency_pred_vec)[:,4]))
                    #guiShared.planningData.vec_planned_vel.append(copy.copy(np.array(pred_vec)[:,2]))
                    #guiShared.planningData.vec_emergency_planned_vel.append(copy.copy(np.array(emergency_pred_vec)[:,2]))

                    #guiShared.planningData.vec_emergency_action.append(emergency_action)
                    guiShared.planningData.append(planningData)
                    guiShared.roll = copy.copy(dataManager.roll)
                    guiShared.real_path = copy.deepcopy(dataManager.real_path)

                    #guiShared.vec_path.append(copy.deepcopy(state['path']))
                    #guiShared.planningData.vec_predicded_path.append([pred[0] for pred in pred_vec])
                    #guiShared.vec_emergency_predicded_path.append([pred[0] for pred in emergency_pred_vec])
                   # guiShared.state = copy.deepcopy(state)
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
           
       