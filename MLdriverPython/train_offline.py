import model_based_algorithm
import matplotlib.pyplot as plt
import numpy as np
#import gym
#import pybullet_envs
import enviroment1
import data_manager1
import hyper_parameters
from model_based_net import model_based_network
import library as lib
import agent_lib as pLib

import os
if __name__ == "__main__": 
    #cd C:\Users\gavri\Desktop\sim_15_3_18
    #cd C:\Users\gavri\Desktop\thesis\ArielUnity - learning2\sim4_5_18
    #sim4_5_18.exe -quit -batchmode -nographics

    #env  = gym.make("HalfCheetahBulletEnv-v0")
    HP = hyper_parameters.ModelBasedHyperParameters()

    dataManager = data_manager1.DataManager(HP.save_file_path,HP.restore_file_path,HP.restore_flag)
    envData = enviroment1.OptimalVelocityPlannerData()#'model_based'
    #net = DDPG_network(envData.observation_space.shape[0],envData.action_space.shape[0],envData.action_space.high[0],\
    #    HP.alpha_actor,HP.alpha_critic,HP.alpha_analytic_actor,HP.alpha_analytic_critic,tau = HP.tau,seed = HP.seed,feature_data_n = envData.feature_data_num, conv_flag = HP.conv_flag)  
    net = model_based_network(envData.observation_space.shape[0],6,HP.alpha)
    if HP.restore_flag:
        net.restore(HP.restore_file_path)





    Replay = pLib.Replay(HP.replay_memory_size)
    #Replay1 = pLib.Replay(HP.replay_memory_size)

    Replay.restore(HP.restore_file_path)

    #predicted_values = pLib.predict_n_next(n,net,state)

    waitFor = lib.waitFor()#wait for "enter" in another thread - then stop = true

    for i in range(1000000):
        if waitFor.stop == [True]:
            break
        rand_state, rand_a, rand_next_state, rand_end, _ = Replay.sample(HP.batch_size)
                            #update neural networs:
                            #pLib.DDQN(rand_state, rand_a, rand_reward, rand_next_state,net,HP)
        pLib.model_based_update(rand_state, rand_a, rand_next_state,rand_end,net,HP)
        if i%100 == 0:
            X = []
            Y_ = []
            for j in range(len(rand_state)):
                X.append([rand_state[j]['vel'], rand_state[j]['steer'],rand_a[j][1],rand_a[j][0]])#action steer, action acc
                Y_.append([rand_next_state[j]['rel_pos'][0],rand_next_state[j]['rel_pos'][1],rand_next_state[j]['rel_ang'],rand_next_state[j]['vel'],rand_next_state[j]['steer'],rand_next_state[j]['roll']])
            print(net.get_loss(X,Y_))
    net.save_model(HP.save_file_path)