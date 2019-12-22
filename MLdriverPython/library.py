# -*- coding: utf-8 -*-
import time
import math
import numpy as np
import classes
import _thread
import c_functions as c_func
import os
import random
import copy

#def init_c_func():

cf = c_func.cFunctions()

    
def clear_screen():
    os.system('cls')


def flat_list(alist):
    return [item for item in alist]

def normalize_value(val,min_val,max_val):
    return val#-1+(val-min_val)*2/(max_val - min_val)

def denormalize_value(val,min_val,max_val):
    return val# 0.5*((max_val - min_val)*val + max_val + min_val)

def normalize(data,norm_vec):

    #try:
    #    n_data = copy.deepcopy(data)
    #    #print("data:",data)
    #    for i in range(len(data[0])):
    #        #dat = np.array(data)[:,i]

    #        min_data = norm_vec[i][0]
    #        max_data = norm_vec[i][1]
    #        if abs((max_data - min_data)) > 0.000001:
    #            for j in range(len(data)):
    #                n_data[j][i] = -1+(data[j][i]-min_data)*2/(max_data - min_data)
    #except:
    #    print("data:",data)
    return data#n_data

def denormalize(n_data,range_vec):
    #data = copy.deepcopy(n_data)
    #for i in range(len(data[0])):
    #    dat = np.array(data)[:,i]

    #    min_data = range_vec[i][0]
    #    max_data = range_vec[i][1]
    #    if abs((max_data - min_data)) > 0.000001:
    #        for j in range(len(data)):
    #            data[j][i] = 0.5*((max_data - min_data)*data[j][i] + max_data + min_data)
    return n_data#data

def dist_vec(A,B):
    return np.linalg.norm(A-B)

def dist(x1,y1,x2,y2):
    tmp = (x2-x1)**2 + (y2- y1)**2
    if tmp > 0:
        return math.sqrt(tmp)
    else:
        return 0.
#def dist(pos1,x2,y2):
#    tmp = (x2-x1)**2 + (y2- y1)**2
#    if tmp > 0:
#        return math.sqrt(tmp)
#    else:
#        return 0.



def comp_curvature(A, B, C):
    tmp = np.cross((B - A), (C - A))
   # print(tmp)
    epsilon = 1e-12
    if abs(tmp[2])<epsilon:#vector in x-y cross vector in z direction
        return 0.0
    diameter = dist_vec(B,C)*dist_vec(A,B)*dist_vec(A,C) / tmp[2]  #R=BC/2sin(A)=BC⋅AB⋅AC/2∥AB×AC∥
    curv = 2./diameter
    return curv

def running_average(data,N):
    if len(data) > 0:
        return np.convolve(data, np.ones((N,))/N, mode='valid')#'valid' same
    else:
        return []
 

def changeZtoY(aList):
    tmp = aList[1]
    aList[1] = aList[2]
    aList[2] = tmp
    return aList
def change_to_rad(alist):
    rad_list=[]
    for angle in alist:
        rad_list.append(math.radians(angle))
    return rad_list

def rotateVec(vec,ang):
    rvec = np.asarray([0.,0.,0.])
    
    rvec[0] = math.cos(ang)*vec[0] - math.sin(ang)*vec[1]
    rvec[1] = math.sin(ang)*vec[0] + math.cos(ang)*vec[1]
    if len(vec) == 3:
        rvec[2] = vec[2]
        return rvec
    else: 
        return rvec[:2]

def to_local(pos, init_pos, init_ang):#pos given in global reference system, convert it to local reference system (located at init_pos in init_ang)
    if len(pos) == 3 or len(init_pos) == 3:
        if len(pos) == 2:
            #pos.append(0)
            pos = np.append(pos,[0])
        if len(init_pos) == 2:
            #pos.append(0)
            init_pos = np.append(init_pos,[0])
    local_pos = np.array(pos) -  np.array(init_pos)
    local_pos = rotateVec(local_pos, init_ang)
    return local_pos.tolist()

def to_global(pos, init_pos, init_ang):#pos given in local reference system (located at init_pos in init_ang), convert it to global reference system 
    if len(pos) == 3 or len(init_pos) == 3:
        if len(pos) == 2:
            #pos.append(0)
            pos = np.append(pos,[0])

        if len(init_pos) == 2:
            #pos.append(0)
            init_pos = np.append(init_pos,[0])
        
    local_rot = rotateVec(pos, -init_ang)
    global_pos = np.array(init_pos) + np.array(local_rot)
    return global_pos.tolist()




def select_target_index(path,velocity = None, index = 0):
    k = 1.
    vel = velocity if velocity is not None else path.velocity[index]
    forward_distance = k*vel
    target_index = index
    while (path.distance[target_index] - path.distance[index]) < forward_distance:
        if target_index >= len(path.distance) - 1:
            break
        target_index += 1
    return target_index

def comp_steer_local(local_target):
    vehicle_lenght = 3.6
    ld2 = local_target[0]**2 + local_target[1]**2
    if ld2 == 0:
        return 0
    curv = 2*local_target[0]/ld2
    steer_ang = -math.atan(curv*vehicle_lenght)
    return steer_ang

def comp_steer(vehicle_position,vehicle_angle,target):
    vehicle_lenght = 3.6
    local_target = to_local(np.asarray(target),np.asarray(vehicle_position),vehicle_angle)#compute target in vehicle reference system
    #ld2 = local_target[0]**2 + local_target[1]**2
    #if ld2 == 0:
    #    return 0
    #curv = 2*local_target[0]/ld2
    #steer_ang = -math.atan(curv*vehicle_lenght)
    return comp_steer_local(local_target)

def comp_steer_target(path,vel, index = 0):
        k = 1.
        min = 2
        max = 6

        if index > len(path.distance)-1:
            index = len(path.distance) - 1
        forward_distance = k*vel#self.desired_path.velocity[index]
        forward_distance = np.clip(forward_distance,min,max)
        target_index = index
        while (path.distance[target_index] - path.distance[index]) < forward_distance:
            if target_index >= len(path.distance) - 1:
                break
            target_index += 1

        #dis_min = path.distance[target_index]
        #if len(path.distance) > target_index+1:
        #    dis_max = path.distance[target_index+1] 
        #    ratio = (forward_distance - dis_min)/(dis_max - dis_min)
        #    print("ratio:",ratio)
        #    x = ratio * path.position[target_index][0] + (1-ratio)*path.position[target_index+1][0]
        #    y = ratio * path.position[target_index][1] + (1-ratio)*path.position[target_index+1][1]
        #else:
            
        x = path.position[target_index][0]
        y = path.position[target_index][1]


        return [x,y]
def find_index_on_path(path,pos,start_index = 0):#return closest index to vehicle
    #distances = [10000. for _ in range(len(path.position))]
    #for i in range(start_index, len(path.position)):
    #        distances[i] = (path.position[i][0] - pos[0])**2 + (path.position[i][1] - pos[1])**2
    #index_min = np.argmin(distances)
    #a =[[1,2],[3,4],[5,6]]
    #t = time.clock()
    #np_path = np.array(path.position[start_index:])
    #print("np.array:",time.clock() - t)
    np_path_tr = np.transpose(path.position[start_index:])
    #print(a_tr)
    distances = np.square( np_path_tr[0] - pos[0])+np.square( np_path_tr[1] - pos[1])
    #print(a_tr)
    return np.argmin(distances)
#def find_index_on_path(path, vehicle, start_index):#return closest index to vehicle
#    distances = [10000. for _ in range(len(path.position))]
#    for i in range(start_index, len(path.position)):
#         distances[i] = (path.position[i].x - vehicle.position[0])**2 + (path.position[i].y - vehicle.position[1])**2
#    index_min = np.argmin(distances)
#    return index_min

def comp_steer_general(path,index,pos,ang,vel):#nearest index on path 
    target = comp_steer_target(path,vel,index = index)
    local_target = to_local(np.asarray(target),np.asarray(pos),ang)#compute target in vehicle reference system
    steer = comp_steer(pos,ang,target)
    #print("target:",target,"local_target:",local_target,"pos:",pos,"pos_on_path:",path.position[index],"ang:",ang,"steer:",steer)
    return steer


#def comp_steer_learn_local(local_target):
#    local_target = local_target[0:2]
#    invert_flag = False
#    if local_target[0] > 0:
#        invert_flag = True
#        local_target[0]*=-1
#    local_target.append(local_target[0]**2/25)
#    local_target.append(local_target[1]**2/25)
#    local_target.append(local_target[0]**3/10000)
#    local_target.append(local_target[1]**3/10000)
#    local_target.append(local_target[0]**4/100000)
#    local_target.append(local_target[1]**4/100000)
#    W = np.array([-0.03772359,-0.03060057,-0.0382355,0.01177276,-0.03024748,0.02575669 ,0.02876469,-0.00775899])#from linear regression
#    #W = np.array([0.00068105,-0.012659 ])#from linear regression
#    b = np.array([0.39522606])
#    steer_ang = np.matmul(local_target, W) + b
#    #steer_ang = local_target[0]*W[0] + local_target[1]*W[1] + b
#    if invert_flag:
#        steer_ang*=-1
#    return steer_ang[0]

def comp_steer_learn(vehicle,target):
    local_target = to_local(np.asarray(target),np.asarray(vehicle.position),vehicle.angle[1])#compute target in vehicle reference system
    return comp_steer_learn_local(local_target)

def read_data(file_name):#, x,y,steer_ang
    with open(file_name, 'r') as f:#append data to the file
        data = f.readlines()
        data = [x.strip().split() for x in data]

    print(data)
    return data
    #for i in range (1,len(path_real.position)):
    #    f.write("%s %s %s %s \n" % (path_real.position[i].x,path_real.position[i].y,path_real.steering[i],path_real.curvature[i]))


def compare_to_compute():
    vh = classes.Vehicle()
    vh.position = [0,0,0]
    vh.angle = [0,0,0]
    file_name = "train_data2.txt"
    data = read_data(file_name)
    data_pos= [[x[0],x[1]] for x in data]
    data_steer = [float(x[2]) for x in data]
    error = [0 for _ in range(len(data_pos))]

    for i in range(1,len(data_pos)):
        error[i] = comp_steer_learn_local([data_pos[i][0], data_pos[i][1],0]) - data_steer[i]
    sum = 0
    for item in error:
        sum+=item**2


    return sum /len(error)

def read_data(file_name):#, x,y,steer_ang
    with open(file_name, 'r') as f:#append data to the file
        data = f.readlines()
        data = [x.strip().split() for x in data]
        results = []
        for x in data:
            results.append(list(map(float, x)))

    return results

def step_now(last_time,step_time):
    t = time.time()
    if t - last_time[0] > step_time:
        if t - last_time[0] > step_time+0.005:
            print("step time too short! deviation:",t - last_time[0] - step_time)
        last_time[0] = t
        return True
    return False

def wait_until_end_step(last_time,step_time,max_time_deviation = 0.005):
    while(True):
        t = time.clock()
        if t - last_time[0] > step_time:
            if t - last_time[0] > step_time + max_time_deviation:
                print("step time too short! deviation:",t - last_time[0] - step_time)
                last_time[0] = t
                return True
            last_time[0] = t
            return False


    while (not lib.step_now(self.last_time,self.step_time)):#wait for the next step (after step_time)
        time.sleep(0.00001)



def path_to_global(path,vh):
    global_path = classes.Path()
    for i in range(len(path.position)):
        #local_rot = rotateVec(np.array(path.position[i].toList()), math.radians(vh.angle[1]))
        #global_pos = np.array(vh.position) + local_rot
        global_pos = to_global(np.array(path.position[i].toList()),np.array(vh.position) ,vh.angle[1])
        point = vector3D()
        point.x =global_pos[0]
        point.y =global_pos[1]
        global_path.position.append(point) 
    return global_path

def createPath(type,size,startX,startY,startAng):
    path = classes.Path()
    startAng = startAng
    if type == "circle":
        lenght = math.pi
        R = size;
        start = np.asarray([startX,startY, 0])
        toCenter = np.asarray([0.,R, 0])
        toCenter = rotateVec(toCenter,startAng)
        cen = start+toCenter
        
        #path.position[0].x = start[0]
        #path.position[0].y = start[1]
        for th in range(int(lenght*100)):
            point = vector3D()
            point.x = cen[0] + R*math.cos(th/100 - startAng)
            point.y = cen[1] + R*math.sin(th/100 - startAng)
            path.position.append(point)    
    return path


def split_to_paths(path,lenght_max):
    paths = []
    
    for i in range(len(path.position)-lenght_max):
        for j in range(i+1,lenght_max+i):
            tmp_path = classes.Path()
            tmp_path.position = path.position[i:j].copy()
            tmp_path.angle = path.angle[i:j].copy()
            tmp_path.velocity = path.velocity[i:j].copy()
            tmp_path.steering = path.steering[i:j].copy()
            paths.append(tmp_path)
    return paths

def save_paths(paths):
    file_name = "local_paths.txt"
    with open(file_name, 'w') as f:#append data to the file
        for path in paths:
            for i in range (len(path.position)):
                f.write("%s \t %s\t %s\t %s\t %s\n" % (path.position[i][0],path.position[i][1],path.angle[i][1],path.velocity[i],path.steering[i]))
            f.write("\n")
    return
def save_data(data,file_name):
    with open(file_name, 'w') as f:#append data to the file
        for dat in data:
            f.write("%s\t %s\t %s\t %s\t %s\n" % (dat.start_steering,dat.end_pos[0],dat.end_pos[1],dat.end_angle,dat.end_steering))
    return

def split_to_data(path,lenght_max):
    data = []
    for i in range(len(path.position)-lenght_max):
        for j in range(i+1,lenght_max+i):
            tmp_data = SteerData()
            tmp_data.start_vel = path.velocity[i]
            tmp_data.start_steering = path.steering[i]
            tmp_data.end_pos = to_local(path.position[j],path.position[i],path.angle[i][1])
            
            tmp_data.end_angle = path.angle[j][1] - path.angle[i][1]
            tmp_data.lenght = path.distance[j] - path.distance[i]
            data.append(tmp_data)
           
    return data

def comp_velocity_limit(path):
    skip = 10
    reduce_factor = 1.0
    if len(path.position)<skip*2:
        velocity_limit = [30 for _ in range(len(path.position))]
        return False
    #to close points cause error in the velocity limit computation (due to diffrention without filtering)
    pos = path.position[0::skip]

    x = [row[0] for row in pos]
    y = [row[1] for row in pos]
    z = [row[2] for row in pos]
    velocity_limit = cf.comp_limit_curve(x,y,z)

    skiped_x = range(0,len(path.position),skip)
    real_x = range(len(path.position))
    path.analytic_velocity_limit = np.interp(np.array(real_x), np.array(skiped_x), np.array(velocity_limit))*reduce_factor

    #plt.plot(np.array(skiped_x), np.array(velocity_limit), 'o')
    #plt.plot(real_x, path.analytic_velocity_limit, '-x')

    #plt.show()
    return False
def comp_velocity_limit_and_velocity(path,skip = 1,reduce_factor = 1.0,init_vel = 0, final_vel = 0):

    if len(path.position)<skip*2:
        velocity_limit = [30 for _ in range(len(path.position))]
        return False
    #to close points cause error in the velocity limit computation (due to diffrention without filtering)
    pos = path.position[0::skip]

    x = [row[0] for row in pos]
    y = [row[1] for row in pos]
    z = [row[2] for row in pos]
    velocity_limit,velocity,dtime_vec,acc_vec,result = cf.comp_limit_curve_and_velocity(x,y,z,init_vel = init_vel/reduce_factor, final_vel = final_vel/reduce_factor)

    skiped_x = range(0,len(path.position),skip)
    real_x = range(len(path.position))
    path.analytic_velocity_limit = np.interp(np.array(real_x), np.array(skiped_x), np.array(velocity_limit))*reduce_factor
    if result == 1:
        path.analytic_velocity = np.interp(np.array(real_x), np.array(skiped_x), np.array(velocity))*reduce_factor

        #dt[i] is the time from point i to i+1, hence the last is not defined hence is 0
        #sum t such that t[i] is the time to reach i, first point is 0 last sum of all dt except the last dt
        t = 0
        analytic_time = []
        for dt in dtime_vec:
            analytic_time.append(t)
            t+=dt
            
        path.analytic_time = np.interp(np.array(real_x), np.array(skiped_x), np.array(analytic_time))#*reduce_factor 
    #path.analytic_acceleration = np.interp(np.array(real_x), np.array(skiped_x), np.array(acc_vec))*reduce_factor
        path.comp_distance()
        for j in range(0,len(path.analytic_velocity)-1): 
            v1 = path.analytic_velocity[j]
            v2 = path.analytic_velocity[j+1]
            d = path.distance[j+1] - path.distance[j]
            if d < 0.01:
                print("_____________________________________________________________________________")
            acc = (v2**2 - v1**2 )/(2*d)#acceleration [m/s^2]
            path.analytic_acceleration.append(acc)
        path.analytic_acceleration.append(path.analytic_acceleration[-1])
    #plt.plot(np.array(skiped_x), np.array(velocity_limit), 'o')
    #plt.plot(real_x, path.analytic_velocity_limit, '-x')

    #plt.show()
    return result


def create_random_path(number,resolution,seed = None):

    if seed != None:
        random.seed(seed)
    max_rand_num = 500#2000#500
    min_rand_num = 0#200#0
    pos = np.array([0.0,0.0,0.0])
    ang = 0
    curvature = 0 
    delta_curvature = 0.007
    path = []
    i=0
    j = 0
    while i < number:
        #des_curvature = random.uniform(-0.12,0.12)
        #if abs (des_curvature) > 0.0001:
        #    max_arc = int((math.pi/(abs(des_curvature)))/0.05)#maximum a 0.5 of a circle
        #    max_rand_num = min(max_rand_num,max_arc)
            #min_arc = int((0.5*math.pi/(abs(des_curvature)))/0.05)#maximum a 0.25 of a circle

        rand_num = random.randint(min_rand_num,max_rand_num)
        des_curvature = random.uniform(-0.12,0.12) #changed order
        #print("num: ",rand_num,"curv: ",des_curvature)
        for j in range(rand_num):
            if curvature < des_curvature:
                curvature += delta_curvature
            if curvature > des_curvature:
                curvature -= delta_curvature
            #print("curv: ", curvature,"des_curvature:",des_curvature)
            ang+= curvature*resolution
            #print(np.array([math.cos(ang),math.sin(ang)]))
            #print(resolution*np.array([math.cos(ang),math.sin(ang)]))
            pos+= resolution*np.array([math.cos(ang),math.sin(ang),0.0])
            path.append(np.copy(pos))
        i+=j


    #pathx = []
    #pathy = []
    #for item in path:
    #    pathx.append(item[0])
    #    pathy.append(item[1])

    #plt.plot(pathx,pathy)
    #plt.show()
    return path[:number]#create longer path - return only needed points

def compute_analytic_path(seed):
    path = classes.Path()
    path.position = create_random_path(9000,0.05,seed = seed)
    comp_velocity_limit_and_velocity(path,skip = 10,reduce_factor = 1.0)
    return path

class _Getch:
#"""Gets a single character from standard input.  Does not echo to the
#screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()


getch = _Getch()



class waitFor:
    def __init__(self):
        self.stop = []
        self.command = []
        self.wait_for()
    def input_thread(self,stop,command):
        stop.append(False)
        command.append(None)
        while True:
            #print("wait___________________________________")
            inp = getch()
            if inp ==  b'\x00':
                continue
            print("input:",inp)
            if inp == b'\r' or inp == '\r':
                stop[0] = True
            else:
                command[0] = inp
               
            time.sleep(0.02)#0.02
        return
    def wait_for(self):
        _thread.start_new_thread(self.input_thread, (self.stop,self.command,))
        return


    
def convert_replay_to_states(replay_memory):
    states = []
    next_states = []
    actions = []
    pos = []
    done = []
    
    for ind in range(len(replay_memory)-1):
        if replay_memory[ind][3] == True or replay_memory[ind+1][4]: # done flag
            print("error - done flag")
            break
        states.append( replay_memory[ind][0])
        actions.append(replay_memory[ind][2])
        next_states.append(replay_memory[ind+1][0])
        pos.append(replay_memory[ind+1][1])
    return states,actions,next_states,pos
###########################################################




