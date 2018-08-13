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

if __name__ == "__main__": 
    #cd C:\Users\gavri\Desktop\sim_15_3_18
    #cd C:\Users\gavri\Desktop\thesis\ArielUnity - learning2\sim4_5_18
    #sim4_5_18.exe -quit -batchmode -nographics

    #env  = gym.make("HalfCheetahBulletEnv-v0")
    HP = HyperParameters()
    #dataManager = data_manager.DataManager(total_data_names = ['total_reward'], special = 1, file = HP.save_name+".txt")#episode_data_names = ['limit_curve','velocity']
    dataManager = data_manager1.DataManager(HP.save_file_path,HP.restore_file_path,HP.restore_flag)
    envData = enviroment1.OptimalVelocityPlannerData()
    net = DDPG_network(envData.observation_space.shape[0],envData.action_space.shape[0],envData.action_space.high[0],\
        HP.alpha_actor,HP.alpha_critic,HP.alpha_analytic_actor,HP.alpha_analytic_critic,tau = HP.tau,seed = HP.seed[0],feature_data_n = envData.feature_data_num, conv_flag = HP.conv_flag)   
    if HP.restore_flag:
        net.restore(HP.restore_file_path)
    #init agent on analytic planner:
    init_nets.init_net_analytic(envData,net,HP.save_file_path,HP.restore_file_path,HP.gamma,create_data_flag = False)
