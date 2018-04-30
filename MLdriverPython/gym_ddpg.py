from library import *  #temp
from classes import Path
import copy
import random
from DQN_net import DQN_network
from DDPG_net import DDPG_network
#from plot import Plot
import json
import policyBasedLib as pLib
import data_manager
import gym
import pybullet_envs
import numpy as np


if __name__ == "__main__": 
    
    ##run simulator: cd C:\Users\student_2\Documents\ArielUnity - learning2\sim_2_1
    #"""
    #cd C:\Users\gavri\Desktop\sim_15_3_18
    #sim15_3_18 -quit -batchmode -nographics
    #"""

    #subprocess.Popen('C:\\Users\\gavri\\Desktop\\sim_15_3_18\\sim15_3_18 -quit -batchmode -nographics')
  
    #pre-defined parameters:
    HP = pLib.HyperParameters()
    ###################
    total_step_count = 0
    steps = [0,0]#for random action selection of random number of steps
    stop = []
    command = []
    wait_for(stop,command)#wait for "enter" in another thread - then stop = true
    #dv = HP.acc * HP.step_time
    #action_space =[-dv,dv]
    #HalfCheetah-v2
    env  = gym.make("HalfCheetahBulletEnv-v0")#''  Pendulum-v0 MinitaurBulletEnv-v0 HalfCheetahBulletEnv-v0 AntBulletEnv-v0 CartPoleBulletEnv-v0
    np.random.seed(HP.seed)
    random.seed(HP.seed)
    env.seed(HP.seed)
    if HP.render_flag:
        env.render()
    env.reset()
    HP.action_space_n = env.action_space.shape[0]
    HP.features_num = env.observation_space.shape[0]
    HP.max_action = env.action_space.high[0]
    total_reward_vec = []
    net = DDPG_network(HP.features_num,HP.action_space_n,HP.max_action,HP.alpha_actor,HP.alpha_critic,tau = HP.tau,seed = HP.seed) 
    
    #net = DQN_network(HP.features_num,len(action_space),HP.alpha_actor,HP.alpha_critic,tau = HP.tau)  
    print("Network ready")
    if HP.restore_flag:
        net.restore(HP.restore_name)
    if not HP.skip_run:
        
        #plot = Plot(total_reward_vec)
        dataManager = data_manager.DataManager(total_data_names = ['total_reward'],  file = HP.save_name+".txt")
        Replay = pLib.Replay(HP.replay_memory_size)
        actionNoise = pLib.OrnsteinUhlenbeckActionNoise(mu=np.zeros(HP.action_space_n))


        for i in range(HP.num_of_runs): #number of runs - run end at the end of the main path and if vehicle deviation error is to big
            if stop == [True]:
                break
            # initialize every episode:
            step_count = 0
            ep_ave_max_q = 0
            reward_vec = []
            last_time = [0]
            #########################
            state = env.reset()
            #first state:
            
            #state = pLib.get_state(pl,local_path,HP.feature_points,HP.distance_between_points)
            for step_count in range(2**32):
                if  stop == [True]:#while not stoped, the loop break if reached the end or the deviation is to big
                    break
                if HP.render_flag:
                    env.render()
    
                #print("steps: ",step_count )
                
                #choose and make action:
                #Q = net.get_Q([state])
                #Pi = net.get_Pi([state])
                #print("velocity1: ",state[0])#,"Q: ",Q)#,"PI: ",Pi)#"velocity2: ",state[1],
              
                noise = actionNoise() * HP.max_action
                #Q = [[net.get_Qa([state],[[0]]),net.get_Qa([state],[[1]])]]
                #print("Q: ",Q)
                a = net.get_actions(np.reshape(state, (1, HP.features_num)))#reshape from array([]) to array([[]])
                a = a[0]
                a +=  noise#one action
                a = np.clip(a,-HP.max_action,HP.max_action)
                #print("action: ", a,"noise: ",noise)

                #get reward:#get next state:
                next_state,reward,done,_ = env.step(a)
                #done = False
                #if step_count > HP.max_episode_length:
                #    done = True
                #print("reward: ",reward)
                reward_vec.append(reward)
                
                #add data to replay buffer:
                Replay.add((state,a,reward,next_state,done))

                if len(Replay.memory) > HP.batch_size:
                    #sample from replay buffer:
                    rand_state, rand_a, rand_reward, rand_next_state,rand_end = Replay.sample(HP.batch_size)
                    
                    #update neural networs:
                    #pLib.DDQN(rand_state, rand_a, rand_reward, rand_next_state,net,HP)
                    pLib.DDPG(rand_state, rand_a, rand_reward, rand_next_state,rand_end,net,HP)
                #print("targetQ:",net.get_targetQ([state]))
                #print("Q:",net.get_Q([state]))
                
                ##actor-critic:
                #next_Q = net.get_Q(next_state)#from this state
                ##print("correcting with action: ",a)
                #Q_[batch_index][a] = reward +HP.gamma*np.max(next_Q)#compute Q_: reward,a - from last to this state, Q - from this state
                #net.Update_Q([state],Q_)

                #Q_last = net.get_Q(last_state[0])#Q for last state for update policy on last state values
                #Q_loss = net.get_Q_loss(state,Q_[batch_index])
                #print("Q_loss: ",Q_loss[0][a])
                #A = Q[a] - sum(Q)/len(Q)
                #net.Update_policy([state],[one_hot_a],[[A]])

              
          
                
                    
                #TD(lambda):
                ##if(len(state_vec) > HP.num_of_TD_steps):
                ##    future_reward = 0
                ##    for k in range(HP.num_of_TD_steps):
                ##        future_reward += reward_vec[k]*HP.gamma**k 
                ##    net.Update_policy([state_vec[-HP.num_of_TD_steps]],[action_vec[-HP.num_of_TD_steps]],[[future_reward]])
                ##    reward_vec.pop(0)#remove first
                ##    value_vec[-HP.num_of_TD_steps] = [future_reward]#the reward is on the last state and action
                ##reward_vec.append(reward)

                ##state_vec.append(last_state[0])#for one batch only
                ##action_vec.append(one_hot_a)
               
                state = next_state

                #save data 
                #dataManager.update_real_path(pl = pl,velocity_limit = local_path.velocity_limit[0])#state[0]
                #dataManager.save_additional_data(reward = reward)#pl,features = denormalize(state,0,30),action = a
                
                if done:
                    break
                
                #end if time
            #end while

            #after episode end:
            #print("mean reward: ",sum(reward_vec)/len(reward_vec))
            total_reward = 0
            #for k,r in enumerate(reward_vec):
            #    total_reward+=r*HP.gamma**k
            total_reward = sum(reward_vec)
            dataManager.add(('total_reward',(total_reward,i)))
            #dataManager.print()
            #plot.update(total_reward_vec)
            print("episode: ", i, " total reward: ", total_reward, "episode steps: ",step_count)
            #print("steps: ",step_count )
            total_step_count += step_count
            #net.update_summaries(total_reward)
            #print("total steps: ",total_step_count )
            #input()
            if (i % HP.save_every == 0 and i > 0): 
                net.save_model(HP.save_name)
            if HP.plot_flag and command == [b'1']:
                dataManager.plot.plot('total_reward')



        #end all:

        net.save_model(HP.save_name)
           
    
