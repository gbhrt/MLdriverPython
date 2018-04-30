

import numpy as np
from library import *  #temp
from classes import Path
import copy
import random
#from policyBasedNet import Network
#from DQN_net import DQN_network
from DDPG_net import DDPG_network
from plot import Plot
import json
import policyBasedLib as pLib
#import data_manager
import subprocess
#import gym
#import pybullet_envs

def train(env,HP,dataManager,seed = None):
    

    #"""
    #cd C:\Users\gavri\Desktop\sim_15_3_18
    #sim15_3_18 -quit -batchmode -nographics
    #"""

    #subprocess.Popen('C:\\Users\\gavri\\Desktop\\sim_15_3_18\\sim15_3_18 -quit -batchmode -nographics')
  
    #pre-defined parameters:
    #HP = pLib.HyperParameters()
    if seed != None:
        HP.seed = seed
    ###################
    total_step_count = 0
    #env = enviroment1.OptimalVelocityPlanner(HP)
    #env  = gym.make("HalfCheetahBulletEnv-v0")
    np.random.seed(HP.seed)
    random.seed(HP.seed)
    env.seed(HP.seed)
    steps = [0,0]#for random action selection of random number of steps
    stop = []
    command = []
    wait_for(stop,command)#wait for "enter" in another thread - then stop = true
    if HP.render_flag:
        env.render()
    #env.reset()
    net = DDPG_network(env.observation_space.shape[0],env.action_space.shape[0],env.action_space.high[0],HP.alpha_actor,HP.alpha_critic,tau = HP.tau,seed = HP.seed)  
    #net = DQN_network(HP.features_num,len(action_space),HP.alpha_actor,HP.alpha_critic,tau = HP.tau)  
    print("Network ready")
    if HP.restore_flag:
        net.restore(HP.restore_name)
    if not HP.skip_run:
        
        #plot = Plot()
        #dataManager = data_manager.DataManager(total_data_names = ['total_reward'],  file = HP.save_name+".txt")
        Replay = pLib.Replay(HP.replay_memory_size)
        actionNoise = pLib.OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.action_space.shape[0]),dt = env.step_time)
        #if not HP.random_paths_flag:
        #    if pl.load_path(HP.path_name,HP.random_paths_flag) == -1:
        #        stop = [1]
      


        for i in range(HP.num_of_runs): #number of runs - run end at the end of the main path and if vehicle deviation error is to big
            if stop == [True]:
                break
            # initialize every episode:
            step_count = 0
            reward_vec = []
            last_time = [0]
            #########################
            state = env.reset()
           
            while  stop != [True]:#while not stoped, the loop break if reached the end or the deviation is to big
                step_count+=1
               
                #choose and make action:
                #Q = net.get_Q([state])
                #Pi = net.get_Pi([state])
                #print("velocity1: ",state[0])#,"Q: ",Q)#,"PI: ",Pi)#"velocity2: ",state[1],
                noise = actionNoise() * env.action_space.high[0]
            
                a = net.get_actions(np.reshape(state, (1, env.observation_space.shape[0])))
                a = a[0]
                a +=  noise#one action
                a = np.clip(a,-env.action_space.high[0],env.action_space.high[0])
                #print("action: ", a,"noise: ",noise)

                next_state, reward, done, info = env.step(a)
                reward_vec.append(reward)

                #add data to replay buffer:
                Replay.add((state,a,reward,next_state,done))
                if len(Replay.memory) > HP.batch_size:
                    #sample from replay buffer:
                    rand_state, rand_a, rand_reward, rand_next_state, rand_end = Replay.sample(HP.batch_size)

                    #update neural networs:
                    #pLib.DDQN(rand_state, rand_a, rand_reward, rand_next_state,net,HP)
                    pLib.DDPG(rand_state, rand_a, rand_reward, rand_next_state,rand_end,net,HP)
               
               
                state = next_state

                #save data 
                #dataManager.update_real_path(pl = pl,velocity_limit = local_path.velocity_limit[0])#state[0]
                #dataManager.save_additional_data(reward = reward)#pl,features = denormalize(state,0,30),action = a
                

                if done:
                    break
                
                #end if time
            #end while

            #after episode end:
            total_reward = 0
            #for k,r in enumerate(reward_vec):
            #    total_reward+=r*HP.gamma**k
            total_reward = sum(reward_vec)
            #dataManager.add(('total_reward',(total_reward,i)))
            dataManager.add(('total_reward',total_reward))
            print("episode: ", i, " total reward: ", total_reward, "episode steps: ",step_count)

            if (i % HP.save_every == 0 and i > 0): 
                net.save_model(HP.save_name)
            if HP.plot_flag and command == [b'1']:
                dataManager.plot.plot('total_reward')
                #dataManager.plot.plot_path_with_features(dataManager,env.distance_between_points,block = True)
                dataManager.plot.plot_path(dataManager.real_path,block = True)
            dataManager.restart()
            #try:
            #    dataManager.comp_rewards(path_num-1,HP.gamma)
            #    dataManager.print_data()
            #except:
            #    print("cannot print data")
            #if HP.plot_flag and command == [b'1']:
            #    plot.close()
            #    plot.plot_path_with_features(dataManager,HP.distance_between_points,block = True)
                #plot.plot_path(dataManager.real_path,block = True)
            
            #dataManager.restart()

        #end all:
    

        net.save_model(HP.save_name)
        
        env.close()
        #del env
        #del HP
        #del net
        #del Replay
        #del actionNoise

        return dataManager.total_data['total_reward']
           
       