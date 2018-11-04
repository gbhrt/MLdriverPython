import DDPG_algorithm
import matplotlib.pyplot as plt
import numpy as np
#import gym
#import pybullet_envs
import enviroment1
import data_manager1
from hyper_parameters import HyperParameters
from DDPG_net import DDPG_network
import init_nets
#import threading
import os
import time
if __name__ == "__main__": 
    #cd C:\Users\gavri\Desktop\sim_15_3_18
    #cd C:\Users\gavri\Desktop\thesis\ArielUnity - learning2\sim4_5_18
    #sim4_5_18.exe -quit -batchmode -nographics

    #cd C:\Users\gavri\Desktop\thesis\ArielUnity - learning2\sim_spring0.25
    #sim_spring0.25.exe -quit -batchmode -nographics 

    #env  = gym.make("HalfCheetahBulletEnv-v0")
    HP = HyperParameters()
    envData = enviroment1.OptimalVelocityPlannerData(mode = "DDPG")
    
    envData.analytic_feature_flag = False
    HP.add_feature_to_action  = False
    HP.analytic_action = False
    HP.restore_flag = False 

    names = ["RL_800_600_400"]#"test_sep_replay_2"
    description = "big net 800_600_400"

    #with_features_new_desing_conv - not good, max 0.8 to many fails
    #not at all
    #low_acc_diff_path no so good, with wheel vel
    #low_acc_diff_path_regular regular features, not work
    HP.reduce_vel = 0.0
    HP.num_of_runs = 10000
    for name in names:
        
        run_data = ["envData.analytic_feature_flag: "+str(envData.analytic_feature_flag), 
                "HP.add_feature_to_action: "+str(HP.add_feature_to_action),
                "roll_feature_flag: "+str(envData.roll_feature_flag),
                "alpha_actor: "+str(HP.alpha_actor),
                "reduce_vel: "+str(HP.reduce_vel),
                description]
        HP.restore_name = name
        HP.save_name = name
        HP.save_file_path = os.getcwd()+ "\\files\\models\\no_friction_hard\\"+HP.save_name+"\\"
        HP.restore_file_path = os.getcwd()+ "\\files\\models\\no_friction_hard\\imitation\\"+HP.restore_name+"\\"

        dataManager = data_manager1.DataManager(HP.save_file_path,HP.restore_file_path,HP.restore_flag)#HP.restore_flag
        dataManager.run_data = run_data
        dataManager.save_run_data()
        net = DDPG_network(envData.observation_space.shape[0],envData.action_space.shape[0],envData.action_space.high[0],\
            HP.alpha_actor,HP.alpha_critic,HP.alpha_analytic_actor,HP.alpha_analytic_critic,tau = HP.tau,seed = HP.seed[0],feature_data_n = envData.feature_data_num, conv_flag = HP.conv_flag)  
        if HP.restore_flag:
            net.restore(HP.restore_file_path)
        #train agent on simulator
        env = enviroment1.OptimalVelocityPlanner(dataManager,mode="DDPG")
        if env.opened:     
            DDPG_algorithm.train(env,HP,net,dataManager)
        #HP.reduce_vel +=0.05
    time.sleep(3)



    #total_rewards_vec = []
    #for i in range(5):
    #    total_rewards_vec.append(DDPG_algorithm.train(env,i))
    #    print(total_rewards_vec)
    #print(total_rewards_vec)

    
    #for i in range(5):
    #    plt.plot(total_rewards_vec[i])
    #mean_rewards = []
    #for j in range(len(total_rewards_vec[0])):
    #    sum = 0
    #    for i in range(5):
    #        sum+=total_rewards_vec[i][j]
    #    mean_rewards.append(sum/5)

    #plt.plot(mean_rewards,'o')

    #plt.show()

