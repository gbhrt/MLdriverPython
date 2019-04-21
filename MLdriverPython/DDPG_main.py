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

def run_train(HP,i = 0):
    if HP.evaluation_flag:
        
        dataManager = data_manager1.DataManager(HP.save_file_path,HP.restore_file_path,restore_flag =False,save_name = 'data_manager_'+str(i))#HP.restore_flag

    else:
        dataManager = data_manager1.DataManager(HP.save_file_path,HP.restore_file_path,HP.restore_flag)#HP.restore_flag
        dataManager.run_data = run_data
        dataManager.save_run_data()
        
            
    net = DDPG_network(envData.observation_space.shape[0],envData.action_space.shape[0],envData.action_space.high[0],\
        HP.alpha_actor,HP.alpha_critic,HP.alpha_analytic_actor,HP.alpha_analytic_critic,tau = HP.tau,seed = HP.seed[0],feature_data_n = envData.feature_data_num, conv_flag = HP.conv_flag)  
    if HP.restore_flag:
        if HP.evaluation_flag:
            if net.restore(HP.restore_file_path,name = 'tf_model_'+str(i)):#cannot restore - return true
                return True
        else:
            net.restore(HP.restore_file_path,name = 'tf_model')
    #train agent on simulator
    env = environment1.OptimalVelocityPlanner(dataManager,env_mode="DDPG")
    if env.opened:     
        DDPG_algorithm.train(env,HP,net,dataManager)
    return False
if __name__ == "__main__": 
    #cd C:\Users\gavri\Desktop\sim_15_3_18
    #cd C:\Users\gavri\Desktop\thesis\ArielUnity - learning2\sim4_5_18
    #sim4_5_18.exe -quit -batchmode -nographics

    #cd C:\Users\gavri\Desktop\sim_14_1_19
    #sim_14_1_19.exe -quit -batchmode -nographics

    #cd C:\Users\gavri\Desktop\thesis\ArielUnity - learning2\sim_spring0.25
    #sim_spring0.25.exe -quit -batchmode -nographics 

    #env  = gym.make("HalfCheetahBulletEnv-v0")
    HP = HyperParameters()
    envData = environment1.OptimalVelocityPlannerData(env_mode = "DDPG")
    
    envData.analytic_feature_flag = False
    HP.add_feature_to_action  = False
    HP.analytic_action = False
    HP.restore_flag = True

    HP.train_flag = False
    HP.always_no_noise_flag = True
    HP.num_of_runs = 1000
    names = ["REVO4"]#,"REVO3","REVO4","REVO5"]#['REVO6',"REVO7","REVO8","REVO9","REVO10"]#
    description = "run without evaluation 15.4.19" 
    HP.evaluation_flag = True
    if HP.evaluation_flag:
        HP.train_flag = False
        HP.always_no_noise_flag = True
        HP.restore_flag = True
        HP.num_of_runs = 100
        HP.save_every_train_number = 10000#saved at every 500, evaluate on a diffferent number

    #with_features_new_desing_conv - not good, max 0.8 to many fails
    #not at all
    #low_acc_diff_path no so good, with wheel vel
    #low_acc_diff_path_regular regular features, not work
    HP.reduce_vel = 0.0
    
    for name in names:

        run_data = ["envData.analytic_feature_flag: "+str(envData.analytic_feature_flag), 
                "HP.add_feature_to_action: "+str(HP.add_feature_to_action),
                "roll_feature_flag: "+str(envData.roll_feature_flag),
                "alpha_actor: "+str(HP.alpha_actor),
                "reduce_vel: "+str(HP.reduce_vel),
                description]
        HP.restore_name = name
        HP.restore_file_path = os.getcwd()+ "\\files\\models\\paper_fix\\"+HP.restore_name+"\\"
        HP.save_name = name
        HP.save_file_path = os.getcwd()+ "\\files\\models\\paper_fix\\"+HP.save_name+"\\"
        if HP.evaluation_flag:
            #nums = [HP.save_every_train_number*j for j in range(1,101)]
            #for i in nums:
            #    if run_train(HP,i):
            if run_train(HP,22500):
                    break
        else:
            run_train(HP)

        #HP.reduce_vel +=0.05
    time.sleep(3)





