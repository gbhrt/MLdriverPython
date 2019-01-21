import copy
import random
import numpy as np
import json
import library as lib
import pathlib
import sys
from math import copysign

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
        #return map(np.array, zip(*samples))
        return map(list, zip(*samples))
    def change_path(self):
        for i in range(len(self.memory)):
            path = self.memory[i][0]
            self.memory[i][0] = path.position

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
            print("cannot save replay buffer:", sys.exc_info()[0])
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

def DDPG(rand_state, rand_a, rand_reward, rand_next_state,rand_end,net,HP,comp_analytic_acceleration = None):
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
    print("critic_loss:",critic_loss)
    
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
def model_based_update(rand_state, rand_a, rand_next_state,rand_end,net,HP,env):
    X,Y_ = env.create_XY_(rand_state, rand_a, rand_next_state)
    net.update_network(X,Y_)
    #print("loss:",net.get_loss(X,Y_))
    return

def comp_abs_pos_ang(rel_pos,rel_ang,abs_pos,abs_ang):
    next_abs_pos = lib.to_global(rel_pos,abs_pos,abs_ang)
    next_rel_ang = abs_ang + rel_ang
    return next_abs_pos,next_rel_ang

def comp_steer_from_next_state(net,env,state,steer_command,acc_command):
    X = env.create_X([state],[[acc_command,steer_command]])
    Y = net.get_Y(X)[0]#not using copy X to Y features, therefore no need to add X
    Y_dict = env.Y_to_Y_dict(Y)
    #abs_pos,abs_ang = comp_abs_pos_ang(Y[0:2],Y[2],[0,0],0)
    rel_pos = [env.denormalize(Y_dict["rel_pos_x"],"rel_pos_x"),env.denormalize(Y_dict["rel_pos_y"],"rel_pos_y")]
    abs_pos,abs_ang = comp_abs_pos_ang(rel_pos,env.denormalize(Y_dict["rel_ang"],"rel_ang"),[0,0],0)
    index = lib.find_index_on_path(state['path'],abs_pos)
    next_steer_command = lib.comp_steer_general(state['path'],index,abs_pos,abs_ang,state['vel_y'])#action steer
    return next_steer_command

def steer_policy(abs_pos,abs_ang,path,index,vel):
    return lib.comp_steer_general(path,index,abs_pos,abs_ang,vel)
def emergency_steer_policy():
    return 0.0
def acc_policy(i,acc_try):
    if i==0:#the first time determent by the current given action
        return acc_try
    return -1.0

def check_stability(env,path,index,abs_pos,roll,roll_var = None,delta_var = 0.0,max_plan_roll = None,max_plan_deviation = None):
    if roll_var is None:
        max_plan_roll*delta_var
    if max_plan_roll is None:
        max_plan_roll = env.max_plan_roll
    if max_plan_deviation is None:
        max_plan_deviation = env.max_plan_deviation
    dev_flag,roll_flag = False,0
    dev_from_path = lib.dist(path.position[index][0],path.position[index][1],abs_pos[0],abs_pos[1])#absolute deviation from the path
    #print("deviation:", dev_from_path)
    delta_var = 0.0#0.002
    roll_var = max_plan_roll*delta_var

    if abs(env.denormalize(roll,'roll'))+roll_var > max_plan_roll: #check the current roll 
        roll_flag = copysign(1,roll)
    if dev_from_path > max_plan_deviation:
        dev_flag = True
        #max_plan_deviation = 10
    return roll_flag,dev_flag
#input: init state and commands 
#output: x dict of the initial state+actions and abs_pos,abs_ang
def initilize_prediction(env,init_state,acc_command,steer_command):
    abs_pos = [0,0]#relative to the local path
    abs_ang = 0
    X = env.create_X([init_state],[[acc_command ,steer_command]])[0]# first action is already executed and known
    return env.X_to_X_dict(X),abs_pos,abs_ang

#input: X dict
#output: X dict after one step
def predict_one_step(net,env,X_dict,abs_pos,abs_ang):
    X = env.dict_X_to_X(X_dict)
    Y = list(net.get_Y([X])[0])#get Y list from X list
    Y_dict = env.Y_to_Y_dict(Y)
    for name in env.copy_Y_to_X_names:
        X_dict[name] += Y_dict[name]
        rel_pos = [env.denormalize(Y_dict["rel_pos_x"],"rel_pos_x"),env.denormalize(Y_dict["rel_pos_y"],"rel_pos_y")]
    abs_pos,abs_ang = comp_abs_pos_ang(rel_pos,env.denormalize(Y_dict["rel_ang"],"rel_ang"),abs_pos,abs_ang)
    #pred = [abs_pos,abs_ang,vel,roll,roll_var]
    return X_dict,abs_pos,abs_ang
#input: X dict
#output: X dict after one step
def predict_n_next1(n,net,env,X_dict,abs_pos,abs_ang,path,acc_try = 1.0,emergency_flag = False,max_plan_roll = None,max_plan_deviation = None,delta_var = 0.0):
    dev_flag,roll_flag = False,0
    pred_vec = []
    roll_var = max_plan_roll*delta_var
    dev_flag,roll_flag = check_stability(env,path,0,abs_pos,X_dict['roll'],roll_var = roll_var,delta_var = 0.0,max_plan_roll = max_plan_roll,max_plan_deviation = max_plan_deviation)
    if dev_flag == False and roll_flag == 0:
        for i in range(0,n):#n times       
            X_dict,abs_pos,abs_ang = predict_one_step(net,env,X_dict,abs_pos,abs_ang)

            roll_var+=delta_var
            #print("roll_var:",roll_var)


            vel = env.denormalize(X_dict['vel_y'],"vel_y")
            index = lib.find_index_on_path(path,abs_pos)

            if emergency_flag:
                steer_command = emergency_steer_policy()
                acc_command = emergency_acc_policy()
            else:
                steer_command = steer_policy(abs_pos,abs_ang,path,index,vel)
                acc_command = acc_policy(i,acc_try)

            X_dict["steer_action"] = env.normalize(steer_command,"steer_action")
            X_dict["acc_action"] = acc_command

            roll = env.denormalize(X_dict["roll"],"roll")

            pred_vec.append([abs_pos,abs_ang,vel,roll,roll_var])

            if vel < 2.0:
                break
            dev_flag,roll_flag = check_stability(env,path,index,abs_pos,X_dict['roll'],roll_var = roll_var,delta_var = 0.0,max_plan_roll = max_plan_roll,max_plan_deviation = max_plan_deviation)
            
            if roll_flag != 0 or dev_flag:
                break
            


            
    #print("end loop time:",time.clock() - env.lt)
    #print("pred:", pred_vec)
    #print("roll:", np.array(pred_vec)[:,3])
    #print("vel:", np.array(pred_vec)[:,2])
   
    #print("end--------------------------")
    return pred_vec,roll_flag,dev_flag

def predict_n_next(n,net,env,init_state,acc_command,steer_command,acc_try = 1.0,emergency_flag = False,max_plan_roll = None,max_plan_deviation = None):
    if max_plan_roll is None:
        max_plan_roll = env.max_plan_roll
    if max_plan_deviation is None:
        max_plan_deviation = env.max_plan_deviation

    X_dict,abs_pos,abs_ang = initilize_prediction(env,init_state,acc_command,steer_command)
    pred_vec = [[abs_pos,abs_ang,init_state['vel_y'],init_state['roll'],0]]
    pred_vec_n,roll_flag,dev_flag = predict_n_next1(n,net,env,X_dict,abs_pos,abs_ang,init_state['path'],acc_try = 1.0,emergency_flag = emergency_flag,max_plan_roll = max_plan_roll,max_plan_deviation = max_plan_deviation,delta_var = 0.0)
    pred_vec+=pred_vec_n
    return pred_vec,roll_flag,dev_flag

#def predict_n_next(n,net,env,init_state,acc_command,steer_command,acc_try = 1.0,emergency_flag = False,max_plan_roll = None,max_plan_deviation = None):
#    if max_plan_roll is None:
#        max_plan_roll = env.max_plan_roll
#    if max_plan_deviation is None:
#        max_plan_deviation = env.max_plan_deviation

#    abs_pos = [0,0]#relative to the local path
#    abs_ang = 0
    
#    if emergency_flag:
#        steer_command = emergency_steer_policy()
#    else:
#        steer_command = steer_policy(abs_pos,abs_ang,init_state['path'],0,init_state['vel_y'])
#    print("after steer_command time:",time.clock() - env.lt)
#    print("current action: ",action, "try action: ",acc_try)
 
#    dev_flag,roll_flag = False,0
#    dev_from_path = lib.dist(init_state['path'].position[0][0],init_state['path'].position[0][1],0,0)#absolute deviation from the path
#    print("deviation:", dev_from_path)
#    delta_var = 0.0#0.002
#    roll_var = max_plan_roll*delta_var
#    if abs(init_state['roll'])+roll_var > max_plan_roll: #check the current roll 
#        roll_flag = copysign(1,init_state['roll'])
#    if dev_from_path > max_plan_deviation:
#        dev_flag = True
#        max_plan_deviation = 10
#    pred_vec = [[abs_pos,abs_ang,init_state['vel_y'],init_state['roll'],roll_var]]#,abs_ang,init_state['vel'],init_state['steer'],init_state['roll']]]
#    X = env.create_X([init_state],[[acc_command ,steer_command]])[0]# first action is already executed and known

#    X_dict = env.X_to_X_dict(X)
    
    
#    if dev_flag == False and roll_flag == 0:
#        for i in range(1,n):#n times       
#            print("before prediction time:",time.clock() - env.lt)
#            X = env.dict_X_to_X(X_dict)
            
#            print("before get Y time:",time.clock() - env.lt)
#            Y = list(net.get_Y([X])[0])#get Y list from X list
#            print("X:",X,"Y:",Y)
#            print("after get Y time:",time.clock() - env.lt)
#            Y_dict = env.Y_to_Y_dict(Y)
#            print("after prediction time:",time.clock() - env.lt)
#            for name in env.copy_Y_to_X_names:
#                Y_dict[name] += X_dict[name]
#                X_dict[name] = Y_dict[name]#env.normalize(Y_dict[name],name)

#            roll_var+=delta_var
#            print("roll_var:",roll_var)
#            print("X_dict:",X_dict,"Y_dict:",Y_dict)
#            X = copy.copy(Y[:len(Y) - 5])#copy the whole relative information (exclude commands, rel pos (2) and rel ang(1))

#            abs_pos,abs_ang = comp_abs_pos_ang(Y[-3:-1],Y[-1],abs_pos,abs_ang)#rel_pos = Y[0:2] rel_ang = Y[2] roll Y[5]
#            rel_pos = [env.denormalize(Y_dict["rel_pos_x"],"rel_pos_x"),env.denormalize(Y_dict["rel_pos_y"],"rel_pos_y")]
#            abs_pos,abs_ang = comp_abs_pos_ang(rel_pos,env.denormalize(Y_dict["rel_ang"],"rel_ang"),abs_pos,abs_ang)#rel_pos = Y[0:2] rel_ang = Y[2] roll Y[5]

#            vel = env.denormalize(Y_dict['vel_y'],"vel_y")
#            index = lib.find_index_on_path(init_state['path'],abs_pos)
#            steer_command =  lib.comp_steer_general(init_state['path'],index,abs_pos,abs_ang,vel)#action steerinit_state['vel_y']

            
#            if emergency_flag:
#                steer_command = emergency_steer_policy()
#                acc_command = emergency_acc_policy()
#            else:
#                steer_command = steer_policy(abs_pos,abs_ang,init_state['path'],index,vel)
#                acc_command = acc_policy(i,acc_try)

#            X_dict["steer_action"] = env.normalize(steer_command,"steer_action")


#            X_dict["acc_action"] = acc_command

#            roll = env.denormalize(Y_dict["roll"],"roll")

#            pred_vec.append([abs_pos,abs_ang,vel,roll,roll_var])
#            if vel < 2.0:
#                break

#            dev_from_path = lib.dist(init_state['path'].position[index][0],init_state['path'].position[index][1],abs_pos[0],abs_pos[1])#absolute deviation from the path
#            print("deviation:", dev_from_path)

            
#            if abs(roll)+roll_var > max_plan_roll: 
#                print("fail rool or dev")
#                roll_flag = copysign(1,roll)
#                break
#            if dev_from_path > max_plan_deviation:
#                dev_flag = True
#                break

            
#    print("end loop time:",time.clock() - env.lt)
#    print("pred:", pred_vec)
#    print("roll:", np.array(pred_vec)[:,3])
#    print("vel:", np.array(pred_vec)[:,2])
   
#    print("end--------------------------")
#    return pred_vec,roll_flag,dev_flag


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




