import model_based_algorithm
import matplotlib.pyplot as plt
import numpy as np
#import gym
#import pybullet_envs
import environment1
import data_manager1
import hyper_parameters
from model_based_net import model_based_network
import train_thread
import os
import shared
import agent_lib as pLib
import time
import agent
import test_net_performance
import test_actions
import cProfile_test
#import simple_env


def run(guiShared,HP,dataManager): 
    #cd C:\Users\gavri\Desktop\sim_15_3_18
    #cd C:\Users\gavri\Desktop\thesis\ArielUnity - learning2\sim4_5_18
    #sim4_5_18.exe -quit -batchmode -nographics

    #env  = gym.make("HalfCheetahBulletEnv-v0")

    """
    start env thread
    start training thread: 
   
    continue main thread for gui
    """

   

   # net = model_based_network(envData.observation_space.shape[0],6,HP.alpha,envData.observation_space.range)
    #net = model_based_network(envData.X_n,envData.Y_n,HP.alpha)

    #if HP.restore_flag:
    #    net.restore(HP.restore_file_path)

    #Replay = pLib.Replay(HP.replay_memory_size)
    #if HP.restore_flag:
    #    Replay.restore(HP.restore_file_path)

    
    #initilize agent:
    envData = environment1.OptimalVelocityPlannerData('model_based')
    if HP.MF_policy_flag:
        Agent = agent.Agent(HP,envData)
    elif HP.program_mode == "test_net_performance":
        Agent = agent.Agent(HP,trans_net_active = True, steer_net_active = False)
    else:
        Agent = agent.Agent(HP,trans_net_active = True, steer_net_active = True if HP.emergency_steering_type == 3 else False)#False

    if HP.program_mode == "train_in_env":
        #initialize environment:
        
        env = environment1.OptimalVelocityPlanner(dataManager,env_mode = "model_based")

        guiShared.max_roll = envData.max_plan_roll
        guiShared.max_time = envData.step_time*envData.max_episode_steps+5#add time for braking
        if env.opened:     
            model_based_algorithm.train(env,HP,Agent,dataManager,guiShared)

    elif HP.program_mode == "test_net_performance":
        test_net_performance.test_net(Agent)

    elif HP.program_mode == "test_actions":
        test_actions.test(Agent)
    elif HP.program_mode == "timing":
        env = environment1.OptimalVelocityPlanner(dataManager,env_mode = "model_based")
        cProfile_test.test(Agent,env)
    #elif HP.program_mode == "simple":
    #    env = environment1.OptimalVelocityPlanner(dataManager,env_mode = "model_based")
    #    cProfile_test.test(Agent,env)
    else:
        print("program_mode unkwnon:",HP.program_mode)


        #trainShared.train = False
    #time.sleep(1.0)
    #trainShared.request_exit = True
    #while not trainShared.exit:
    #    time.sleep(0.1)
    #print("exit from train thread")
    guiShared.exit = True


