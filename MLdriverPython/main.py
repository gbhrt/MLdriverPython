import DDPG_algorithm
import matplotlib.pyplot as plt
import numpy as np
import gym
import pybullet_envs
import enviroment1
import data_manager
from hyper_parameters import HyperParameters


if __name__ == "__main__": 
    #cd C:\Users\gavri\Desktop\sim_15_3_18
    #sim15_3_18 -quit -batchmode -nographics

    #env  = gym.make("HalfCheetahBulletEnv-v0")
    HP = HyperParameters()
    dataManager = data_manager.DataManager(total_data_names = ['total_reward'], special = 1, file = HP.save_name+".txt")#episode_data_names = ['limit_curve','velocity']

    env = enviroment1.OptimalVelocityPlanner(dataManager)
    if env.opened:
        DDPG_algorithm.train(env,HP,dataManager)

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

