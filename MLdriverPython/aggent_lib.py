import copy
import random
import numpy as np
import json
import library as lib
import pathlib

import time

class Replay:
    def __init__(self,replay_memory_size):
        self.memory_size = replay_memory_size
        self.memory = []
    def add(self,data):
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        self.memory.append(data)
    def sample(self,batch_size):
        samples = random.sample(self.memory,np.clip(batch_size,0,len(self.memory)))
        return map(np.array, zip(*samples))
    def save(self,path,name = "replay_memory"):
        print("save replay buffer...")
        try:
            path += "replay_memory\\"
            pathlib.Path(path).mkdir(parents=True, exist_ok=True) 
            file_name = path + name + ".txt"
            with open(file_name, 'w') as f:
                json.dump(self.memory,f)
            print("done.")

        except:
            print("cannot save replay buffer")
            return



    def restore(self,path,name = "replay_memory"):
        print("restore replay buffer...")
        try:
            path += "replay_memory\\"
            pathlib.Path(path).mkdir(parents=True, exist_ok=True) 
            file_name = path + name + ".txt"
            with open(file_name, 'r') as f:
                self.memory = json.load(f)
            print("done.")
        except:
            print('cannot restore replay buffer')
        return

        
def DDQN(rand_state, rand_a, rand_reward, rand_next_state,net,HP):
    #compute target Q:
    rand_next_Q = net.get_Q(rand_next_state)
    rand_next_a = np.argmax(rand_next_Q,axis = 1)#best action from next state according to Q network
    rand_next_targetQ = net.get_targetQ(rand_next_state)
    rand_targetQ = []
    for i in range(len(rand_state)):
        #rand_targetQ.append(rand_reward[i] + HP.gamma*np.max(rand_next_Q[i])) #DQN
        rand_targetQ.append(rand_reward[i] + HP.gamma*rand_next_targetQ[i][rand_next_a[i]])#DDQN

    #update nets:
    net.Update_Q(rand_state,rand_a,rand_targetQ)
    net.update_target()
    return

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

def DDPG(rand_state, rand_a, rand_reward, rand_next_state,rand_end,net,HP):
    #compute target Q:
    
    rand_next_a = net.get_target_actions(rand_next_state)#action from next state
    
    ##vec0 = [[0] for _ in range(len(rand_next_state))]
    ##vec1 = [[1] for _ in range(len(rand_next_state))]
    #Q0 = net.get_Qa(rand_next_state,vec0)
    #3Q1 = net.get_Qa(rand_next_state,vec1)
    
    ##rand_next_Q = [[Q0[i][0],Q1[i][0]] for i in range(len(Q0))]

    ##rand_next_a = np.argmax(rand_next_Q,axis = 1)#best action from next state according to Q network
    ##rand_next_a = [[item] for item in rand_next_a]
    rand_next_targetQa = net.get_targetQa(rand_next_state,rand_next_a)#like in DQN
    rand_targetQa = []
    for i in range(len(rand_state)):
        if rand_end[i] == False:
            rand_targetQa.append(rand_reward[i] + HP.gamma*rand_next_targetQa[i])#DQN  
        else:
            rand_targetQa.append([rand_reward[i]])
    #update critic:
    net.Update_critic(rand_state,rand_a,rand_targetQa)#compute Qa(state,a) and minimize loss (Qa - targetQa)^2
    Qa = net.get_Qa(rand_state,rand_a)
    critic_loss = net.get_critic_loss(rand_state,rand_a,rand_targetQa)
    
    
    #update actor
    pred_action = net.get_actions(rand_state)#predicted action from state
    #print("actions: ",pred_action)
    #print("params: ",net.get_actor_parms())
    #print("grads: ",net.get_actor_grads(rand_state,pred_action))
    #print("norm grads: ",net.get_norm_actor_grads(rand_state,pred_action))
    #print("grads: ",net.get_neg_Q_grads(rand_state,pred_action))
    
    net.Update_actor(rand_state,pred_action)
    net.update_targets()
    return critic_loss, Qa#temp

def choose_action(action_space,Pi,steps = None,epsilon = 0.1):
    if random.random() < epsilon:
        a = random.randint(0,len(action_space) - 1)#random.randint(0,(len(action_space.data) - 1))
        print("random a: ",a)
    else:
        a = np.argmax(Pi)
        print("best a: ",a)
    
    #action = np.random.choice(list_of_candidates, number_of_items_to_pick, p=probability_distribution)

    #choose a random action for the next random steps:
    #if steps[0] == 0:
    #    steps[0] = random.randint(1,10)
    #    steps[1] = random.randint(0,len(action_space) - 1)
    #    print("choose: steps: ",steps[0],"action: ",steps[1])
    #a = steps[1]
    #steps[0] -= 1
    #print("steps: ",steps[0])

    #choose action from propbilities of Pi:
    #if random.random() < epsilon:
    #    a = random.randint(0,len(action_space) - 1)#random.randint(0,(len(action_space.data) - 1))
    #    print("random a: ",a)
    
    #else:
    #    rand = random.random()
    #    prob = 0
    #    for i in range(len(action_space)):
    #        prob += Pi[i]
    #        if rand < prob:
    #            a =  i
    #            break

    #always the highest probability:
    #a = np.argmax(Pi)

    return a




