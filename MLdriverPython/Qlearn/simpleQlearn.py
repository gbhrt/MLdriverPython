from planner import Planner
import numpy as np
import random
import time
from library import wait_for
import math
import copy


class ActionSpace:
    def __init__(self):
        self.data = []
        self.n = 0
        return
    def comp_lenght(self):
        self.n =len(self.data)
  
    

class StateSpace:
    def __init__(self):
        self.data = [[0],[0]]
        self.n = 0
        return
    def comp_lenght(self):
        for item in self.data:
            self.n+=len(item)
    #ef get_index(self):
def find_nearest(array,value):

    idx = np.searchsorted(array, value, side="left")

    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):

        return array[idx-1]
    else:

        return array[idx]
def find_nearest_index(array,value):

    idx = np.searchsorted(array, value, side="left")

    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):

        return idx-1
    else:

        return idx
def get_state(pl,state_space):
    pos = pl.find_index_on_path(0)
    vel = find_nearest_index(state_space.data[1],pl.simulator.vehicle.velocity)
    return pos, vel

def comp_best(Q):
    best = copy.copy(Q[:,:,0])
    #print("best: ", best)
    for i in range(len(state_space.data[0])):
        for j in range(len(state_space.data[1])):
            best[i,j] = np.argmax(Q[i,j,:])#Q[i,j,np.argmax(Q[i,j,:])]

    #best = copy.copy(Q[:,0])
    #for j in range(len(state_space.data[1])):
    #    best[j] = np.argmax(Q[j,:])#Q[i,j,np.argmax(Q[i,j,:])

    best = np.rot90(best)
    print("best: \n", best)
    return best

def save_Q(file,Q):
     with open (file,"w") as f:
         if isinstance(Q, list) or type(Q).__module__ == np.__name__:
            for item in Q:
                if isinstance(item, list) or type(Q).__module__ == np.__name__:
                    for it in item:
                        if isinstance(it, list) or type(Q).__module__ == np.__name__:
                            for i in it:
                                f.write('%f ' %i)
                            f.write('\n') 
                        else:
                            f.write('%f ' %it) 
                            f.write('\n') 
                else:
                    f.write('%f ' %item)
                    f.write('\n')
         else:
            f.write('%f ' %Q) 

     f.close()
     #for item in Q:
     #    for ite in item:
     #       logFile.write('%f ' %ite) 
     #    logFile.write('\n') 
     #logFile.close()
     return

def restore_Q(file,Q):
    with open(file, 'r') as f:#append data to the file
        data = f.readlines()
        data = [x.strip().split() for x in data]
        
        results = copy.copy(Q)
        
        for i in range(len(Q)):
            for j in range(len(Q[0])):
                results[i][j] = (list(map(float, data[i*len(Q[0]) + j])))

        #for x in data:
        #    results.append(list(map(float, x)))

        #for i in len(Q):
        #    Q[i] = 
    return np.array(results)

if __name__ == "__main__": 

    #Q = np.zeros([9,5,3])
    ###Q = np.random.rand(len(state_space.data[1]),action_space.n)
    ###Q = np.zeros([len(state_space.data[1]),action_space.n])
    #Q = restore_Q("Q3.txt",Q)
    #print (Q)

    pl = Planner()
    action_space = ActionSpace()
    state_space = StateSpace()
    #pl.simulator.set_address("10.2.1.111",5007)# if run in linux # "10.0.17.74"
    pl.start_simple()
    pl.create_path1()

    print("Planner ready")

    epsilon = 0.3
    gamma = 0.95
    alpha = 0.3
    num_of_runs = 10000
    step_time = 0.5

    
    print("start")
    stop = []
    wait_for(stop)#wait for "enter" in annother thread - then stop = true

    #pl.simulator.reset_position()

    max_dist = pl.desired_path.distance[len(pl.desired_path.distance)-1]# maximum distance to target
 
    res_dist = 1.# resolution (1/m)
    max_vel = 5.# maximum velocity
    res_vel = 1.# resolution (1/(m/s))

    state_space.data[0] = copy.copy(pl.desired_path.distance)#[x/res_dist for x in range (math.floor(max_dist*res_dist))]
    state_space.data[1] = [x/res_vel for x in range (math.floor(max_vel*res_vel))]
    state_space.comp_lenght()
    dv = 1.
   
    action_space.data = [-dv,0.,dv]
    action_space.comp_lenght()

    #Q = np.zeros([len(state_space.data[0]),len(state_space.data[1]),action_space.n])
    max_reward  = max_vel
    Q = np.full((len(state_space.data[0]),len(state_space.data[1]),action_space.n), max_reward)
    #Q = np.random.rand(len(state_space.data[1]),action_space.n)
    #Q = np.zeros([len(state_space.data[1]),action_space.n])
    #Q = restore_Q("Q3.txt",Q)
    print (Q)

    #comp_best(Q)
    for i in range(num_of_runs): #number of runs
        if stop:
            break

        #state_value =  pl.dist_from_target() #pl.get_state_space_index()
        #s = pl.find_index_on_path(0)
        pos,vel = get_state(pl,state_space)
        if pos == 0 and vel > 0:
            print("??")
        #s = pos*len(state_space.data[0]) + vel
        done = False
        total_reward = 0
        while not done and not stop: #until game over, or user stop
             if random.random() < epsilon:
                a = random.randint(0,2)#random.randint(0,(len(action_space.data) - 1))
                print("random a: ",a)
             else:
                 a = np.argmax(Q[pos,vel,:])
                 #a = np.argmax(Q[vel,:])
                 print("best a: ",a)
             #step:
             
             des_vel = pl.simulator.vehicle.velocity + action_space.data[a]
             #print("des vel: ", pl.simulator.vehicle.velocity," + ", action_space.data[a]," = ",des_vel)
             pl.simulator.send_drive_commands(des_vel,0)
             
             time.sleep(step_time)
             pl.simulator.get_vehicle_data()
             
             pos_next,vel_next = get_state(pl,state_space)
             
             #reward 1 if recived the end and velocity is 0
             #if pos_next == len(state_space.data[0]) - 1 and vel_next <= 1:
             #   r = 1
             #else:
             #   r = 0

             #reward2 - 
             if pos_next == len(state_space.data[0]) - 1:
                 r = max_vel - vel_next
             elif pos_next > pos:
                 r = 1
             else:
                 r = 0

             #reward3 - 
             if pos_next == len(state_space.data[0]) - 1:
                 r = max_vel - vel_next
             elif pos_next > pos:
                 r = 1
             else:
                 r = 0

             total_reward += r
             print("pos: ", pos, " vel: ", vel, "pos_next: ", pos_next," vel_next: ", vel_next, " r: :", r)
             #end step
      
             Q[pos,vel,a] = Q[pos,vel,a] + alpha*(r + gamma * np.max(Q[pos_next,vel_next,:]) - Q[pos,vel,a])
             #Q[vel,a] = Q[vel,a] + alpha*(r + gamma * np.max(Q[vel_next,:]) - Q[vel,a])
             comp_best(Q)
             #print(np.around(Q,decimals=2))
            # print("Q: ",Q,"\n")

             #if vel_next == len(state_space.data[1]) - 1:
             if pos_next == len(state_space.data[0]) - 1:
                print("total reward: ",total_reward)
                done = True
                if i % 20 == 0:
                    pl.simulator.reset_position()
                    time.sleep(1)
                pl.restart()
                pl.create_path1()
                
             
             pos,vel = pos_next,vel_next
             

    pl.stop_vehicle()
    save_Q("Q_reward3.txt",Q)
    pl.end()
      #Q[s,a] = Q[s,a] + alpha*(r + gamma * np.max(Q[s_next,:]) - Q[s,a])
             #print(Q,"\n")
          #if random.random() < epsilon:
          #    a = random.choice(pl.action_space)
          #else:
          #    a = np.argmax(Q[s,:])
          #s_n, r, done, _ = env.step(a)
          #Q[s,a] = Q[s,a] + alpha*(r + gamma * np.max(Q[s_n,:]) - Q[s,a])
          #s = s_n
