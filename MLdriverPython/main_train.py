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
if __name__ == "__main__": 
    #cd C:\Users\gavri\Desktop\sim_15_3_18
    #cd C:\Users\gavri\Desktop\thesis\ArielUnity - learning2\sim4_5_18
    #sim4_5_18.exe -quit -batchmode -nographics

    #cd C:\Users\gavri\Desktop\thesis\ArielUnity - learning2\sim_spring0.25
    #sim_spring0.25.exe -quit -batchmode -nographics 

    #env  = gym.make("HalfCheetahBulletEnv-v0")
    HP = HyperParameters()
    envData = enviroment1.OptimalVelocityPlannerData()#mode = 'DDPG'
    """
    finished:
    regular1 - long 900 episodes
    regular3 - 700 - from 300 continued training, seems like a problem from there
    add_acc1 - 700 episodes

    regular3 - reached 260 episodes
    in home computer- add_acc_feature2 - finished. add_acc_feature2. add_acc2

    """
    envData.analytic_feature_flag = False
    HP.add_feature_to_action  = False
    HP.analytic_action = False
    HP.restore_flag = False
    #names = ["regular13"]#600 steps
    #names = ["add_acc3","add_acc5","add_acc7","add_acc9"]
   # names = ["‏‏reward_shaping1_continue"]
    #names = ["regular3"]
    #names = ["add_acc_feature_uni"]#not bad
    #names = ["test"]
    #names = ["regular_uninoise"]
    
    #names = ["add_acc_feature_uni_test_critic"]

    names = ["regular2","regular4","regular6","regular8","regular10"]
    HP.num_of_runs = 500
    for name in names:
        HP.restore_name = name
        HP.save_name = name
        HP.save_file_path = os.getcwd()+ "\\files\\models\\final3\\"+HP.save_name+"\\"
        HP.restore_file_path = os.getcwd()+ "\\files\\models\\final3\\"+HP.restore_name+"\\"

        dataManager = data_manager1.DataManager(HP.save_file_path,HP.restore_file_path,HP.restore_flag)

        net = DDPG_network(envData.observation_space.shape[0],envData.action_space.shape[0],envData.action_space.high[0],\
            HP.alpha_actor,HP.alpha_critic,HP.alpha_analytic_actor,HP.alpha_analytic_critic,tau = HP.tau,seed = HP.seed,feature_data_n = envData.feature_data_num, conv_flag = HP.conv_flag)  
        if HP.restore_flag:
            net.restore(HP.restore_file_path)
        #train agent on simulator
        env = enviroment1.OptimalVelocityPlanner(dataManager)
        if env.opened:     
            DDPG_algorithm.train(env,HP,net,dataManager)



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

