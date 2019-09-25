import DDPG_algorithm
import matplotlib.pyplot as plt
import numpy as np
#import gym
#import pybullet_envs
import environment1
import data_manager1
from hyper_parameters import HyperParameters
from DDPG_net import DDPG_network
import init_nets
#import threading
import os
import time
import random

def run_train(HP,i = 0):
    global_train_count = 0
    if HP.evaluation_flag:
        dataManager = data_manager1.DataManager(HP.save_file_path,HP.restore_file_path,restore_flag =False,save_name = 'data_manager_'+str(i))

    else:
        dataManager = data_manager1.DataManager(HP.save_file_path,HP.restore_file_path,HP.restore_flag)
        dataManager.run_data = run_data
        dataManager.save_run_data()
        
            
    net = DDPG_network(envData.observation_space.shape[0],envData.action_space.shape[0],\
        HP.alpha_actor,HP.alpha_critic,HP.alpha_analytic_actor,HP.alpha_analytic_critic,tau = HP.tau,seed = HP.seed[0],feature_data_n = envData.feature_data_num, conv_flag = HP.conv_flag)  
    if HP.restore_flag:
        if HP.evaluation_flag:
            if net.restore(HP.restore_file_path,name = 'tf_model_'+str(i)):#cannot restore - return true
                return True
        else:#try to restore the last one
            nums = [HP.save_every_train_number*j for j in range(1,50)]
            found_flag = False
            for i in nums:
                if not net.restore(HP.restore_file_path,name = 'tf_model_'+str(i)):
                    found_flag = True
                elif found_flag:
                    global_train_count = i-HP.save_every_train_number
                    break
            print("train,global_train_count:",global_train_count)
                

    #train agent on simulator
    env = environment1.OptimalVelocityPlanner(dataManager,env_mode="DDPG")
    if env.opened:     
        DDPG_algorithm.train(env,HP,net,dataManager,global_train_count = global_train_count)
    return False

#def DDPG_main():
if __name__ == "__main__":


     #cd C:\Users\gavri\Desktop\sim_15_3_18
    #cd C:\Users\gavri\Desktop\thesis\ArielUnity - learning2\sim4_5_18
    #sim4_5_18.exe -quit -batchmode -nographics

    #cd C:\Users\gavri\Desktop\sim_14_1_19
    #sim_14_1_19.exe -quit -batchmode -nographics

    #cd C:\Users\gavri\Desktop\sim 29_4_19
    #sim_28_4_19.exe -quit -batchmode -nographics

    #cd C:\Users\gavri\Desktop\thesis\ArielUnity - learning2\sim_spring0.25
    #sim_spring0.25.exe -quit -batchmode -nographics 

    #cd C:\Users\gavri\Desktop\sim_14_1_19_no_trail
    #sim_14_1_19.exe -quit -batchmode -nographics 


    #env  = gym.make("HalfCheetahBulletEnv-v0")
    HP = HyperParameters()
    envData = environment1.OptimalVelocityPlannerData(env_mode = "DDPG")
    """
    on each set, run training until 100000, if exist- pass.
    and then, run evaluation, if exist - pass
    """
    names_vec = []
    #REVO10 85k - 0 fails, 1.13 reward 

    #names_vec.append([['REVO+F5'],'REVO+F',70000])
    #names_vec.append([['REVO+A3'],'REVO+A',90000])
    #names_vec.append([['REVO10'],'REVO',85000])
    #names_vec.append([['VOD_long2'],"VOD",0])

    names_vec.append([['REVO+A1'],'REVO+A',15000])

    #names_vec.append([['REVO6','REVO7','REVO8','REVO9','REVO10'],'REVO'])#
    #names_vec.append([['REVO+A1','REVO+A2','REVO+A3','REVO+A4','REVO+A5'],'REVO+A'])
    #names_vec.append([['REVO+F1','REVO+F2','REVO+F3','REVO+F4','REVO+F5'],'REVO+F'])
    #names_vec.append([['REVO+FA1','REVO+FA2','REVO+FA3','REVO+FA4','REVO+FA5'],'REVO+FA'])

    #names_vec.append([['VOD_0175'],"VOD"])#["VOD_00","VOD_002","VOD_004","VOD_006","VOD_008","VOD_01","VOD_012","VOD_014","VOD_016","VOD_018"]
    #random.seed(12)
    #HP.seed = random.sample(range(1000),1000)
    #print("seed",HP.seed)
    #HP.test_same_path = True

    for names,method,training_num in names_vec:
        HP.analytic_action = False
        if method =='REVO':
            envData.analytic_feature_flag = False
            HP.add_feature_to_action  = False
        if method =='REVO+A':
            envData.analytic_feature_flag = False
            HP.add_feature_to_action  = True
        if method =='REVO+F':
            envData.analytic_feature_flag = True
            HP.add_feature_to_action  = False
        if method =='REVO+FA':
            envData.analytic_feature_flag = True
            HP.add_feature_to_action  = True

        if method =='VOD':
            envData.analytic_feature_flag = False
            HP.add_feature_to_action  = False
            HP.analytic_action = True
            HP.evaluation_flag = True
            #reduce_vec = [0.02*i for i in range(1,10)]
            reduce_vec = [0.0]
            HP.train_flag = False
            HP.always_no_noise_flag = True
            HP.restore_flag = False
            HP.num_of_runs = 100
            HP.save_every_train_number = 500000000
            
            for name,reduce in zip(names,reduce_vec):
                HP.restore_name = name
                HP.restore_file_path = os.getcwd()+ "/files/models/paper_fix/"+HP.restore_name+"/"
                HP.save_name = name#"save_movie"#
                HP.save_file_path = os.getcwd()+ "/files/models/paper_fix/"+HP.save_name+"/"
                HP.reduce_vel = reduce
                run_data = []
                run_train(HP)
            #break

        description = method 
        run_data = ["envData.analytic_feature_flag: "+str(envData.analytic_feature_flag), 
            "HP.add_feature_to_action: "+str(HP.add_feature_to_action),
            "reduce_vel: "+str(HP.reduce_vel),
            "seed: "+str(HP.seed),
            description]




        for evalutaion_flag in [True]:  #(False,True) 
            


            #HP.test_same_path = True
            #HP.run_same_path = True
            #HP.evaluation_every = 10000000
            #HP.save_every_train_number = 2500

            
            
            #seeds = [1111,1112,1113,1114,1115]
            #seeds = [1546]
            #["VOD_014","VOD_016","VOD_024"]
            #names = ["VOD_002","VOD_004","VOD_006","VOD_008","VOD_01","VOD_012","VOD_014","VOD_016","VOD_018"]#"same_REVO1"] ['REVO6',"REVO7","REVO8","REVO9","REVO10"]#
            #["same_REVO1","same_REVO2","same_REVO3","same_REVO4","same_REVO5"]
            #reduce_vec = [0.02*i for i in range(1,10)]#[0.14,0.16,0.24,0,0,0,0,0,0,0,0]
            

            HP.evaluation_flag = evalutaion_flag#True

            if HP.evaluation_flag:
                HP.train_flag = False
                HP.always_no_noise_flag = True
                HP.restore_flag = True
                HP.num_of_runs = 100
                HP.save_every_train_number = 5000#saved at every 500, evaluate on a different number
            else:
                HP.train_flag = True
                HP.always_no_noise_flag = False
                HP.restore_flag = True
                HP.num_of_runs = 1000
                HP.save_every_train_number = 2500
    
            #for name,seed,reduce_vel in zip(names,seeds,reduce_vec):
            for name in names:
                #HP.seed = [seed]
                #HP.reduce_vel = reduce_vel#18,20,22

        
                HP.restore_name = name
                HP.restore_file_path = os.getcwd()+ "/files/models/paper_fix/"+HP.restore_name+"/"
                HP.save_name = name#"save_movie"#
                HP.save_file_path = os.getcwd()+ "/files/models/paper_fix/"+HP.save_name+"/"

                if HP.evaluation_flag:
                    #run_train(HP)
                    if training_num is  None:
                        nums = [HP.save_every_train_number*j for j in range(1,50)]
                    else:
                        nums = [training_num]
                    for i in nums:
                        print("num:",i)
                        
                        dataManager_tmp = data_manager1.DataManager(HP.save_file_path,HP.restore_file_path,restore_flag = True,restore_name = 'data_manager_'+str(i))
                        #if not dataManager_tmp.error and len(dataManager_tmp.episode_end_mode) >= HP.num_of_runs-10:
                            
                        #    print(name,'data_manager_'+str(i)+' exist')
                        #    continue
                        print("HP.num_of_runs:",HP.num_of_runs)
                        print("evaluation on episode:",i)
                        if run_train(HP,i):
                            print(name,"cannot restore:",'tf_model_'+str(i))
                            continue
                
                        #HP.restore_flag = True
                else:
                    run_train(HP)

            #HP.reduce_vel +=0.05
            time.sleep(3)







