import model_based_algorithm
import matplotlib.pyplot as plt
import numpy as np
#import gym
#import pybullet_envs
import enviroment1
import data_manager1
import hyper_parameters
from model_based_net import model_based_network
import train_thread
import os
import shared
import agent_lib as pLib
import time

def run(guiShared): 
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
    envData = enviroment1.OptimalVelocityPlannerData('model_based')
    #net = DDPG_network(envData.observation_space.shape[0],envData.action_space.shape[0],envData.action_space.high[0],\
    #    HP.alpha_actor,HP.alpha_critic,HP.alpha_analytic_actor,HP.alpha_analytic_critic,tau = HP.tau,seed = HP.seed,feature_data_n = envData.feature_data_num, conv_flag = HP.conv_flag)  

   # net = model_based_network(envData.observation_space.shape[0],6,HP.alpha,envData.observation_space.range)
    net = model_based_network(envData.X_n,envData.Y_n,HP.alpha,envData.observation_space.range)

    if HP.restore_flag:
        net.restore(HP.restore_file_path)

    Replay = pLib.Replay(HP.replay_memory_size)
    if HP.restore_flag:
        Replay.restore(HP.restore_file_path)


    #train agent on simulator
    env = enviroment1.OptimalVelocityPlanner(dataManager,mode = "model_based")

    trainShared = shared.trainShared()
    if HP.train_flag:
        trainTread = train_thread.trainThread(net,Replay,HP,env,trainShared)
        trainTread.start()
    if env.opened:     
        model_based_algorithm.train(env,HP,net,Replay,dataManager,trainShared,guiShared)
    trainShared.train = False
    time.sleep(1.0)
    trainShared.exit = True
    guiShared.exit_program = True


