import model_based_algorithm
import matplotlib.pyplot as plt
import numpy as np
#import gym
#import pybullet_envs
import enviroment1
import data_manager1
import hyper_parameters
from model_based_net import model_based_network

import os
def run(shared): 
    #cd C:\Users\gavri\Desktop\sim_15_3_18
    #cd C:\Users\gavri\Desktop\thesis\ArielUnity - learning2\sim4_5_18
    #sim4_5_18.exe -quit -batchmode -nographics

    #env  = gym.make("HalfCheetahBulletEnv-v0")

    """
    start env thread
    start training thread: 
   
    continue main thread for gui
    """

    HP = hyper_parameters.ModelBasedHyperParameters()

    dataManager = data_manager1.DataManager(HP.save_file_path,HP.restore_file_path,HP.restore_flag)
    envData = enviroment1.OptimalVelocityPlannerData()#'model_based'
    #net = DDPG_network(envData.observation_space.shape[0],envData.action_space.shape[0],envData.action_space.high[0],\
    #    HP.alpha_actor,HP.alpha_critic,HP.alpha_analytic_actor,HP.alpha_analytic_critic,tau = HP.tau,seed = HP.seed,feature_data_n = envData.feature_data_num, conv_flag = HP.conv_flag)  
    net = model_based_network(envData.observation_space.shape[0],6,HP.alpha)
    if HP.restore_flag:
        net.restore(HP.restore_file_path)
    #train agent on simulator
    env = enviroment1.OptimalVelocityPlanner(dataManager)
    if env.opened:     
        model_based_algorithm.train(env,HP,net,dataManager,shared)


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

