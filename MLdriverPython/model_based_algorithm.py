import time
import numpy as np
import library as lib
import classes
import copy
import random
import predict_lib
#import subprocess
import math
import os
import direct_method
import comp_error

def train(env,HP,Agent,dataManager,guiShared,const_seed_flag = False,global_train_count = 0): #the global train number in MB is the step number, training is async, for compability to other methods

    #subprocess.Popen('C:/Users/gavri/Desktop/sim_15_3_18/sim15_3_18 -quit -batchmode -nographics')
    #pre-defined parameters:
    #Agent.start_training()

    seed = HP.seed[0]
    first_flag = True
    print("start training")
    env_state = env.reset(seed = seed)
    acc,steer = 0.0,0.0
    state = Agent.get_state(env_state)
    acc,steer,planningData = Agent.comp_action(state,acc,steer)
    print("init acc")
    #time.sleep(5)
    ###################
    total_step_count = 0
    random_action_flag = True
    #env = environment1.OptimalVelocityPlanner(HP)
    #env  = gym.make("HalfCheetahBulletEnv-v0")
    np.random.seed(HP.seed[0])
    random.seed(HP.seed[0])
    env.seed(HP.seed[0])
    steps = [0,0]#for random action selection of random number of steps
    waitFor = lib.waitFor()#wait for "enter" in another thread - then stop = true


    #global_train_count = 0
    

    lt = 0#temp
    last_time_stamp = 0
    for i in range(Agent.trainHP.num_of_runs): #number of runs - run ends at the end of the main path and if vehicle deviation error is to big
        if guiShared is not None: request_exit = guiShared.request_exit
        else: request_exit = False
        if waitFor.stop == [True] or request_exit:
            break
        
        # initialize every episode:
        if not HP.run_same_path and HP.evaluation_flag and const_seed_flag:
            if i<len(HP.seed):
                seed = HP.seed[i]
            else:
                print("given seed list is too short! take random seed")
                seed = int.from_bytes(os.urandom(8), byteorder="big")
        else:
            seed = int.from_bytes(os.urandom(8), byteorder="big")
        print("seed:", seed)


        violation_count = 0
        step_count = 0
        reward_vec = []
        t1 = 0

        env_state = env.reset(seed = seed)
        if env_state == "error":
            print("error at env.reset")
            i-=1
            continue
        state = Agent.get_state(env_state)
        if guiShared is not None:
            guiShared.restart()
        if env.error:
            i-=1
            continue
        #episode_start_time = time.time()
        acc,steer = 0.0,0.0
        acc,steer,planningData= Agent.comp_action(state,acc,steer)#for the first time (required because the first time is longer)

        env.pl.init_timer()
        if guiShared is not None: request_exit = guiShared.request_exit
        else: request_exit = False
            
        while  waitFor.stop != [True] and not request_exit:#while not stoped, the loop break if reached the end or the deviation is to big          
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
                if guiShared is not None:
                    with guiShared.Lock:
                        #guiShared.planningData.append(planningData)
                        guiShared.roll = copy.copy(dataManager.roll)
                        guiShared.real_path = copy.deepcopy(dataManager.real_path)
                        guiShared.update_data_flag = True

                next_state, reward, done, info = env.step(acc)#input the estimated next actions to execute after delta t and getting next state
                
            else:#not HP.analytic_action
                acc,steer,planningData = Agent.comp_action(state,acc,steer)#env is temp  ,roll_flag,dev_flag 
                roll_flag, _ = Agent.Direct.check_stability2(state.Vehicle,state.env,Agent.trainHP.max_plan_deviation)
                violation_count+=roll_flag
                #print("acc:",acc,"steer:",steer)
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
                if HP.gui_flag:
                    with guiShared.Lock:
                        guiShared.planningData.append(planningData)
                        guiShared.roll = copy.copy(dataManager.roll)
                        guiShared.real_path = copy.deepcopy(dataManager.real_path)
                        #print("path:",guiShared.real_path.position)
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
            #print('info:',info)
            if abs(env.pl.simulator.vehicle.input_time - last_time_stamp-env.step_time) <0.01 and not time_step_error:
                time_error = False
            else:
                time_error = True
            if not HP.evaluation_flag:
                Agent.add_to_replay(state,acc,steer,done,time_error,time_step_error)# fail
            #print("replay:",Agent.Replay.memory[-1])
            last_time_stamp = env.pl.simulator.vehicle.input_time
            
            global_train_count+=1
            if global_train_count % HP.save_every_train_number == 0:# or global_train_count == 1:#uncomment for saving the initial state for evaluation process
                #print("break in global_train_count % HP.save_every_train_number == 0 and global_train_count > 0:")
                break
            
            #state = copy.deepcopy(next_state)
            
            #print("copy state time:",time.clock() - env.lt)
            #if not HP.analytic_action:
            #    acc,steer = copy.copy(next_acc), copy.copy(next_steer)
            if done:
                #print("break in done")
                break
        if global_train_count>HP.max_steps and not HP.evaluation_flag:
            #print("break in global_train_count>HP.max_steps and not HP.evaluation_flag")
            break
            #end if time
        #end while

        #after episode end:
        env.stop_vehicle_complete()
        if first_flag:
            Agent.start_training()#after first episode to avoid long first training in the middle of driving
            first_flag = False
       #stop at the end of the episode for training
        if not HP.evaluation_flag and HP.pause_for_training and global_train_count > 10 :# global_train_count >10  tmp to ensure that training is already started
            train_count_at_end = 5000
            current_traint_count = Agent.trainShared.train_count
            while Agent.trainShared.train_count - current_traint_count < train_count_at_end:
                time.sleep(5)
                env.pl.stop_vehicle()#for maintain connection to simulator (prevent timeout)
                
        if Agent.trainHP.update_var_flag:
            compute_var_done = [False]
            compErrorThread = comp_error.compError(Agent,compute_var_done)#step_count-1
            compErrorThread.start()
            while not compute_var_done[0]:
                env.pl.stop_vehicle()#for maintain connection to simulator (prevent timeout)
                time.sleep(0.1)
        
        #Agent.update_episode_var(step_count-1)
        
        total_reward = sum(reward_vec)
        if not HP.gym_flag: #and not HP.noise_flag:
            #if step_count<env.max_episode_steps and info[0] == 'ok':
            #    print("error - step_count<env.max_episode_steps and info[0] == ok")
            #    input()
            dataManager.violation_count.append(violation_count)
            dataManager.episode_end_mode.append(info[0])
            dataManager.rewards.append(total_reward)
            dataManager.lenght.append(step_count)
            dataManager.add_run_num(i)
            dataManager.add_train_num(global_train_count)
            dataManager.path_seed.append(env.path_seed)#current used seed (for paths)
            dataManager.update_paths()
            relative_reward = dataManager.comp_relative_reward1(env.pl.in_vehicle_reference_path,last_ind,last_tim)
            dataManager.relative_reward.append(relative_reward)
            if guiShared is not None:
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
        
        #if not HP.run_same_path:
        #    seed = int.from_bytes(os.urandom(8), byteorder="big")
        #else:#not needed 
        #    seed = HP.seed[0]

        #if (i % HP.zero_noise_every == 0 and i > 0) or HP.always_no_noise_flag:
        #    HP.noise_flag = False
        #    if HP.test_same_path:
        #        seed = HP.seed
        #if (i % HP.evaluation_every == 0 and i > 0) or test_path_ind != 0 or HP.always_no_noise_flag or evaluate:
        #    #HP.noise_flag = False
        #    evaluation_flag = True
        #    if HP.test_same_path:
        #        test_path_ind +=1
        #        seed = HP.seed[test_path_ind]
        #        print("seed:",seed)
        #        if test_path_ind >= len(HP.seed):
        #            test_path_ind = 0

        #Agent.copy_nets()

        if (global_train_count % HP.save_every_train_number == 0 or global_train_count == 1) and not HP.evaluation_flag:# and global_train_count > 0):
            if global_train_count == 1:
                HP.net_name = 'tf_model_'+str(0)
                Agent.var_name = 'var_'+str(0)
            else:
                HP.net_name = 'tf_model_'+str(global_train_count)
                Agent.var_name = 'var_'+str(global_train_count)

            Agent.save_nets()
            Agent.save_var()
        if (i % HP.save_every == 0 and i > 0): 
            dataManager.save_data()
        if HP.plot_flag and waitFor.command == [b'1']:
            dataManager.plot_all()
        if guiShared is not None:
            while guiShared.pause_after_episode_flag:
                time.sleep(0.1)
                env.pl.stop_vehicle()#for maintain connection to simulator (prevent timeout)
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
           
       
