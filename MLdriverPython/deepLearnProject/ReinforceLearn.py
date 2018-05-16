from planner import Planner
import numpy as np
from library import *  #temp
from classes import Path
import copy
import random
from ReinforceNet import Network

class StateSpace:
    def __init__(self):
        self.data = []
        self.n = 0
        self.features_num = 0
        return
    def comp_lenght(self):
        for item in self.data:
            self.n+=len(item)

def copy_path(path,start,end):#return path from start to end
    cpath = Path()
    if end > len(path.position):
        end = len(path.position)

    for i in range(start,end):
        cpath.position.append(path.position[i])
        cpath.angle.append(path.angle[i])
        cpath.velocity.append(path.velocity[i])
    return cpath
def choose_points(local_path,number,skip):#return points from the local path, choose a point every "skip" points
    
    n = 0
    points = []
    while n < len(local_path.position):
        points.append(local_path.position[n][0])
        points.append(local_path.position[n][1])
        if len(points) == number * 2:
            break
        n+=skip
    dis = 2
    ang = local_path.angle[-1][1]
    while len(points) < number * 2:
        points.append(local_path.position[-1][0] + dis* math.sin(ang))
        points.append(local_path.position[-1][1] + dis* math.cos(ang))
    return points

def get_state(pl,local_path,points,target_index):
    state = []
    if len(local_path.position) >= 2:
        local_ang = (math.atan2(local_path.position[1][1] - local_path.position[0][1],local_path.position[1][0] - local_path.position[0][0]) -math.pi/2)
        print("local angle: ",local_ang,"angle: ",local_path.angle[0][1])

    steer = pl.simulator.vehicle.steering #current steering of the vehicle
    #vel = pl.simulator.vehicle.velocity
    state.append(steer)

    state += choose_points(local_path,points,30)
    #path_ang = local_path.angle[0][1]
    #state.append(path_ang)
    #point1 = math.copysign(dist(local_path.position[0][0],local_path.position[0][1],0,0),local_path.position[0][0])#distance from path
    #state.append(point1)

    #for i in range (points):
    #    state.append(local_path.position[i][0])#x
    #    state.append(local_path.position[i][1])#y
    #state.append(local_path.position[target_index][0])#x
    #state.append(local_path.position[target_index][1])#y

    return state

def get_reward(local_path,state,max_deviation,last_dev):    
    dev = dist(local_path.position[0][0],local_path.position[0][1],0,0)
    #print("dev: ",dev)
   
    if dev >= max_deviation:#last_dev[0]:
        reward = -100
    else:
         reward = - dev#deviation max_deviation 
    #elif dev > 1:#last_dev[0]:
    #    reward = -1
    #else:
    #    reward = 0.
    #last_dev[0] = dev
    return reward
def choose_action(epsilon,action_space_len,Q):
    if random.random() < epsilon:
        a = random.randint(0,action_space_len - 1)#random.randint(0,(len(action_space.data) - 1))
        print("random a: ",a)
    else:
        a = np.argmax(Q)
        print("best a: ",a)
    return a
def comp_path(pl,main_path,main_index,num_of_points):
    pl.desired_path = copy_path(main_path,main_index,main_index+num_of_points)#choose 100 next points from vehicle position
    pl.simulator.send_path(pl.desired_path)
    local_path = pl.path_to_local(pl.desired_path)#translate path in vehicle reference system
    pl.desired_path.comp_distance()
    return local_path 
def mirror_state(state):
    m_state = [-x for x in state]
    return m_state
def mirror_Q(Q):
    m_Q = [Q[1],Q[0]]
    return m_Q

if __name__ == "__main__": 
    #run simulator: cd C:\Users\student_2\Documents\ArielUnity - learning2\sim_2_1
    #sim_2_1_17 -quit -batchmode -nographics
    #pre-defined parameters:
    feature_points = 3 #not neccecery at the beginning also 1 is good
    features_num = 1+feature_points*2 #vehicle state (now just steering) + points on the path (x,y)
    epsilon = 0.2
    gamma = 0.99
    num_of_runs = 100000
    step_time = 0.2#0.02
    alpha = 1e-7 #learning rate
    max_deviation = 3 # [m] if more then maximum - end episode 
    batch_size = 5

    restore_flag = True
    visualized_points = 200 #how many points show on the map - just visualy
    vel = 2.#constant velocity
    omega = 0.7 # [rad/s] maximum steering angular velocity need to be more then maximum steering angular velocity in real
    res = 1
    path_name = "paths/path3.txt"    #long random path: path3.txt  #long straight path: straight_path.txt
    
    ###################

    pl = Planner()



    main_path = pl.read_path_data(path_name)#get a path at any location
   
    dang = omega * step_time /res
    action_space =[-dang,dang]

    last_state = [0 for _ in range(batch_size)]
    Q_corrected = [0 for _ in range(batch_size)]

    net = Network(alpha,features_num,len(action_space))   
    print("Network ready")

    if restore_flag:
        net.restore("model7.ckpt")###################################

    main_path.set_velocity(vel)
    main_path.comp_angle()
    max_index = len(main_path.position)
    stop = []
    command = []
    wait_for(stop,command)#wait for "enter" in another thread - then stop = true
    for i in range(num_of_runs): #number of runs - run end at the end of the main path and if vehicle deviation error is to big
        if stop:
            break
        index = 0
        main_index = 0
        last_t = 0
        last_dev = [0]
        batch_index= - 1 #first time last Q and last state not exist,  add to batch just from next step
        first_time = True
        main_path_trans = pl.path_tranformation_to_local(main_path)# transform to vehicle position and angle
        mean_dev = 0
        count = 0
        ##initialize Q
        #local_path = comp_path(pl,main_path_trans,main_index,visualized_points)#compute local path and global path(inside the planner)
        ##temp - for comparation
        #target_index = pl.select_target_index(index)
        #steer_ang1 = comp_steer(pl.simulator.vehicle,pl.desired_path.position[target_index])#target in global
        ##end temp
        #state[batch_index] = get_state(pl,local_path,feature_points,target_index)



        #Q = net.get_Q(state)

        ##make action:
        #a = choose_action(epsilon,action_space.n,Q)#choose action 
        #steer_ang = pl.simulator.vehicle.steering + action_space.data[a]

        #print("steer_ang: ",steer_ang1," steer_ang learn: ",steer_ang )
        
        #pl.simulator.send_drive_commands(vel,steer_ang) #send commands
        
         
        #pl.update_real_path(state[batch_index][1])
        

        while not stop:#while not stoped, the loop break if reached the end or the deviation is to big
            t = time.time()
            if t - last_t > step_time:
                if t - last_t > step_time+0.002:
                    print("step time too short!")
                last_t = t
                
                pl.simulator.get_vehicle_data()#read data after time step from last action
                index = pl.find_index_on_path(0)
                main_index += index
                local_path = comp_path(pl,main_path_trans,main_index,visualized_points)#compute local path and global path(inside the planner)

                #print("main index: ", main_index, " index: ",index)
                #temp - for comparation
                target_index = pl.select_target_index(index)
                steer_ang1 = comp_steer(pl.simulator.vehicle,pl.desired_path.position[target_index])#target in global
                #end temp
                if not first_time:
                    last_state[batch_index] =copy.copy(state)#copy current state to list of last states
                state = get_state(pl,local_path,feature_points,target_index)
                dev = dist(local_path.position[0][0],local_path.position[0][1],0,0)
                if not first_time:
                    reward = get_reward(local_path,state,max_deviation,last_dev)
                    Q_corrected[batch_index] = np.copy(Q)
               
                    print("__________________________________________")
                #print("deviation: ",state[1]," steering: ",state[0],"x: ",state[2]," y: ",state[3]," reward: ",reward)#
                    
                    print("deviation: ",dev," steering",state[0],"path angle: ",state[1]," reward: ",reward)#
                #W1,b1 = net.get_par()
                #print("W0 - before: ",W1)
                #print("b - before: ",b1)
                    print("Q - before: ",Q)

                Q = net.get_Q(state)
                #if state[1] < 0:#deviation negative
                #    Q = mirror_Q(Q)
                print ("state: ",state)

                if not first_time:
                    print("correcting with action: ",a)
                    Q_corrected[batch_index][a] = reward +gamma*np.max(Q)
                    print("Q - corrected: ",Q_corrected)
                    
                #W1,b1 = net.get_par()
                #print("W0 - after: ",W1)
                #print("b - after: ",b1)
                
                #Q_last = net.get_Q(last_state)#temp
                #print ("last_state: ",last_state)
                #print("Q - after update (last step):",Q_last)
                #print("Q - after:",Q)
                

            
                
                
                #make action:
                a = choose_action(epsilon,len(action_space),Q)#choose action 
                steer_ang = pl.simulator.vehicle.steering + action_space[a]
                if not first_time:
                    loss = net.get_loss(last_state[batch_index],Q_corrected[batch_index])
                    print("loss: ",loss[0][a])
                print("steer_ang - computed: ",steer_ang1," steer_ang estimated: ",steer_ang )
                pl.simulator.send_drive_commands(vel,steer_ang) #send commands
                pl.update_real_path(state[2])
                batch_index += 1
                if batch_index >= batch_size:
                    if not first_time:
                        net.update_sess(last_state,Q_corrected)#update session on the mini-batch
                    batch_index = 0
                count+= 1
                mean_dev = (dev + (count - 1)*mean_dev)/count
                print("mean_dev ", mean_dev)

                first_time = False
                
                if main_index > max_index-1 or dev > max_deviation: #check if end of the episode
                    #net.save_model()
                    #time.sleep(1)
                    if i % 20 == 0:
                        pl.simulator.reset_position()
                        pl.stop_vehicle()
                        #net.save_model()
                        time.sleep(1)
                    pl.restart()#stop vehicle, and initalize real path
                    break
                
            #end if time
        #end while
    pl.stop_vehicle()
    net.save_model("model7.ckpt")#model4.ckpt - LINEAR, LINE. model5.ckpt - net, line - good, model6.ckpt - 3 points model7.ckpt -very good
    pl.end()

