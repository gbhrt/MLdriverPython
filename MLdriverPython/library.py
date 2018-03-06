import time
import math
import numpy as np
from classes import *
import _thread
import c_functions as c_func
import matplotlib.pyplot as plt

epsilon = 0.000000001

cf = c_func.cFunctions()
    
    
class vector3D:
    def __init__(self):
         self.x = 0
         self.y = 0
         self.z = 0
    x = 0
    y = 0
    z = 0
    def toList(self):
        return [self.x, self.y,self.z]


def input_thread(stop):
    input()
    stop.append(True)
    return
def wait_for(stop):
    _thread.start_new_thread(input_thread, (stop,))
    return

def dist_vec(A,B):
    return np.linalg.norm(A-B)

def dist(x1,y1,x2,y2):
    tmp = (x2-x1)**2 + (y2- y1)**2
    if tmp > 0:
        return math.sqrt(tmp)
    else:
        return 0.



def compCurvature(A, B, C):
    tmp = np.cross((B - A), (C - A))
   # print(tmp)
    if abs(tmp[2])<epsilon:#vector in x-y cross vector in z direction
        return 0.0
    diameter = dist_vec(B,C)*dist_vec(A,B)*dist_vec(A,C) / tmp[2]  #R=BC/2sin(A)=BC⋅AB⋅AC/2∥AB×AC∥
    curv = 2./diameter
    return curv

def printToFile(fileName, pos,backPos):#"C:\\Users\\gavri\\Desktop\\logFile.txt"
     logFile = open(fileName,"w")
     logFile.write("count   pos backPos \n")
     for i in range(100):
         logFile.write('%d %f %f %f %f %f %f \n' %(i,pos[i].x,pos[i].y,pos[i].z, backPos[i].x,backPos[i].y,backPos[i].z)) 
     logFile.close()
     return

 

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

#def rotateVec(vec,cen,ang):
#    rvec[0] = math.cos(ang)*(vec[0]-cen[0]) - math.sin(ang)*(vec[1] - cen[1]) + cen[0]
#    rvec[1] = math.sin(ang)*(vec[0]-cen[0]) + math.cos(ang)*(vec[1] - cen[1]) + cen[1]
#    return rvec

def to_local(pos, initPos, initAng):#pos given in global reference system, convert it to local reference system (located at init_pos in init_ang)
    if len(pos) == 2:
        pos.append(0)
    localPos = np.array(pos) -  np.array(initPos)
    localPos = rotateVec(localPos, initAng)
    return localPos.tolist()

def to_global(pos, initPos, initAng):#pos given in local reference system (located at init_pos in init_ang), convert it to global reference system 
    if len(pos) == 2:
        pos.append(0)
    local_rot = rotateVec(pos, -initAng)
    global_pos = np.array(initPos) + np.array(local_rot)
    return global_pos.tolist()

def end_check(pos, radius):
    if pos[0]**2 + pos[1]**2 > radius**2 or pos[1] < -0.2:
        return True
    return False

def read_path_from_file(file_name):
    path = Path()
    with open(file_name, 'r') as f:
        data = f.readlines()
        data = [x.strip().split() for x in data]
        results = []
        for x in data:
            results.append(list(map(float, x)))
        for item in results:
            point = vector3D()
            point.x =item[0]
            point.y =item[1]
            point.z =0
            path.position.append(point)
    return path

def find_index_on_path(path, vehicle, start_index):#return closest index to vehicle
    distances = [10000. for _ in range(len(path.position))]
    for i in range(start_index, len(path.position)):
         distances[i] = (path.position[i].x - vehicle.position[0])**2 + (path.position[i].y - vehicle.position[1])**2
    index_min = np.argmin(distances)
    return index_min

def select_target_index(path,vehicle,index):
    k = 1.
    forward_distance = k*path.velocity[index]
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

def comp_steer(vehicle,target):
    vehicle_lenght = 3.6
    local_target = to_local(np.asarray(target),np.asarray(vehicle.position),vehicle.angle[1])#compute target in vehicle reference system
    ld2 = local_target[0]**2 + local_target[1]**2
    if ld2 == 0:
        return 0
    curv = 2*local_target[0]/ld2
    steer_ang = -math.atan(curv*vehicle_lenght)
    return steer_ang

def comp_steer_learn_local(local_target):
    local_target = local_target[0:2]
    invert_flag = False
    if local_target[0] > 0:
        invert_flag = True
        local_target[0]*=-1
    local_target.append(local_target[0]**2/25)
    local_target.append(local_target[1]**2/25)
    local_target.append(local_target[0]**3/10000)
    local_target.append(local_target[1]**3/10000)
    local_target.append(local_target[0]**4/100000)
    local_target.append(local_target[1]**4/100000)
    W = np.array([-0.03772359,-0.03060057,-0.0382355,0.01177276,-0.03024748,0.02575669 ,0.02876469,-0.00775899])#from linear regression
    #W = np.array([0.00068105,-0.012659 ])#from linear regression
    b = np.array([0.39522606])
    steer_ang = np.matmul(local_target, W) + b
    #steer_ang = local_target[0]*W[0] + local_target[1]*W[1] + b
    if invert_flag:
        steer_ang*=-1
    return steer_ang[0]

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
    vh = Vehicle()
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
        if t - last_time[0] > step_time+0.002:
            print("step time too short!")
        last_time[0] = t
        return True
    return False



def path_to_global(path,vh):
    global_path = Path()
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
    path = Path()
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
            tmp_path = Path()
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
###########################################################



def readVehicleData(comm):
    vh = Vehicle()
    comm.readData()
    dataType = comm.deserialize(1,int)        
    vh.position = changeZtoY(comm.deserialize(3,float))
    vh.position[2] = 0.
    vh.angle = change_to_rad(comm.deserialize(3,float))
    vh.backPosition = changeZtoY(comm.deserialize(3,float))
    vh.steering = comm.deserialize(1,float)
    return vh

def getVehicleData(comm):
    dataType = 0
    comm.serialize(dataType)
    comm.sendData()
    return readVehicleData(comm)

def sendPath(comm,path_des):
    dataType = 2
    comm.serialize(dataType)
    comm.serialize(len(path_des.position))
    for i in range(len(path_des.position)):
        comm.serialize(path_des.position[i].x)
    for i in range(len(path_des.position)):
        comm.serialize(path_des.position[i].y)
        #comm.serialize(path_des.position[i].z)
    comm.sendData()
    return

def sendDriveCommands(comm,velocity,steering):#send drive commands to simulator
    dataType = 1
    comm.serialize(dataType)
    comm.serialize(velocity)
    comm.serialize(steering)
    comm.sendData()
    return




def create_path_in_run(comm):
    resolution  = 0.1
    v = 0
    steerAng = 0
    sendDriveCommands(comm,v,steerAng)#send commands
    initState = readVehicleData(comm)#read data (respose from simulator to commands)
    time.sleep(1.)
    path_real = Path()
    def run_const():
        vh = getVehicleData(comm)#request data from simulator
        vh.position = to_local(np.asarray(vh.position),np.asarray(initState.position),initState.angle[1])
        point = vector3D()
        point.x =vh.position[0]
        point.y =vh.position[1]
        point.z =vh.position[2]
        path_real.position.append(point)
        time.sleep(resolution)

    vel = 5
    steerAng  = 0
    sendDriveCommands(comm,vel,steerAng)#send commands
    for i in range(50):
        run_const()
    steerAng  = 0.5
    sendDriveCommands(comm,vel,steerAng)#send commands
    for i in range(50):
        run_const()
    steerAng  = -0.5
    sendDriveCommands(comm,vel,steerAng)#send commands
    for i in range(50):
        run_const()
    stop_vehicle(comm,steerAng)
    file_name = "path1.txt"
    with open(file_name, 'w') as f:#append data to the file
        for i in range (1,len(path_real.position)):
            f.write("%s\t %s\n" % (path_real.position[i].x,path_real.position[i].y))
    return path_real



def run_constant_ang(comm,path_real,vel,steerAng,data_radius,resolution):
    v = 0
    sendDriveCommands(comm,v,steerAng)#send commands
    initState = readVehicleData(comm)#read data (respose from simulator to commands)
    time.sleep(1.)
       
    #vel = input("insert vel and ang: ")#get data from user
    #if vel == 'e':
    #    comm.endConnection()
    #    break
    #steerAng = input()
       
    sendDriveCommands(comm,vel,steerAng)#send commands
    vh = readVehicleData(comm)#read data (respose from simulator to commands)
    vh.position = to_local(np.asarray(vh.position),np.asarray(initState.position),initState.angle[1])

    i = 0
    while not end_check(vh.position,data_radius): #save points along the path

        vh = getVehicleData(comm)#request data from simulator

        vh.position = to_local(np.asarray(vh.position),np.asarray(initState.position),initState.angle[1])
            
        point = vector3D()
        point.x =vh.position[0]
        point.y =vh.position[1]
        point.z =vh.position[2]


        path_real.position.append(point)
            
        #print("pos: ",vh.position)#,"poslist: ",path_real.position[i].toList())
            
        curv = 0
        if i >= 3:
            curv = compCurvature(np.asarray(path_real.position[i-2].toList()),np.asarray(path_real.position[i-1].toList()),np.asarray(path_real.position[i].toList()))
            print(" curv: ", curv," pos: ",path_real.position[i-2].toList()," ", path_real.position[i-1].toList()," ", path_real.position[i-2].toList())
            #print ("steering: ", vh.steering)
        path_real.curvature.append(curv)
        path_real.steering.append(steerAng)
        time.sleep(resolution)
        i+=1 # 

    stop_vehicle(comm,steerAng)
    return


def stop_vehicle(comm,steerAng):
    vel = 0 # stop vehicle
    sendDriveCommands(comm,vel,steerAng)#send commands
    vh = readVehicleData(comm)#read data (respose from commands)
    time.sleep(1.)#wait for stop


def test_point(comm,vel,steerAng,data_radius,x,y):
    resolution = 0.1
    v = 0
    sendDriveCommands(comm,v,steerAng)#send commands
    initState = readVehicleData(comm)#read data (respose from simulator to commands)
    time.sleep(1.)

    path_real = Path()
    run_constant_ang(comm,path_real,vel,steerAng,data_radius,resolution)
    distances = []
    for item in path_real.position:
        distances.append((item.x - x)**2 + (item.y - y)**2) 
    min_dist = min(distances)
    print("Minimum distances: ",min_dist)
    return min_dist

def create_data_set(comm,data_radius,angStep,resolution):

    file_name = "train_data.txt"
   
    runTime = 2.
    vel = 2 # m/s constant speed - start speed
    steerAng = 0.
    max_vel = 25.
    vel_step = 5.

    with open(file_name, 'w') as f:#write data to the file - clear the file
        f.write("pos list: \n")

    while vel < max_vel:
        while steerAng < math.pi/4:#check all steering angles from 0 (straight) to 90 deg (not possible - stop at maximum steering angle)
            path_real = Path()
            run_constant_ang(comm,path_real,vel,steerAng,data_radius,resolution)

            print("end path")
            steerAng += angStep
            #print("pos list------------------------------------------------------------------------: \n")
            #for item in path_real.position:
            #    print(item.toList(),'\n')

            with open(file_name, 'a') as f:#append data to the file
                for i in range (1,len(path_real.position)):
                    f.write("%s %s %s \n" % (path_real.position[i].x,path_real.position[i].y,path_real.steering[i]))
        vel+=vel_step
    return




    


def run_on_path(comm, path, vehicle):
    path.comp_path_parameters()
    vel = 2
    path.set_velocity(vel)
    max_index = len(path.position)
    index = find_index_on_path(path, vehicle,0)
    while index < max_index-1:#while not reach the end
        target_index = select_target_index(path,vehicle,index)
        steer_ang = comp_steer_learn(vehicle,[path.position[target_index].x,path.position[target_index].y,0])

        print("index: ",index," target_index: ",target_index, " steer_ang: ", steer_ang)
        sendDriveCommands(comm,vel,steer_ang)#send commands
        vehicle = readVehicleData(comm)#read data (respose from simulator to commands)

        index = find_index_on_path(path, vehicle,index)#asume index increase

    stop_vehicle(comm,0)
    return


def comp_velocity_limit(path):
    skip = 20#to close points cause error in the velocity limit computation (due to diffrention without filtering)
    pos = path.position[0::skip]

    x = [row[0] for row in pos]
    y = [row[1] for row in pos]
    z = [row[2] for row in pos]
    velocity_limit = cf.comp_limit_curve(x,y,z)

    skiped_x = range(0,len(path.position),skip)
    real_x = range(len(path.position))
    path.velocity_limit = np.interp(np.array(real_x), np.array(skiped_x), np.array(velocity_limit))

    #plt.plot(np.array(skiped_x), np.array(velocity_limit), 'o')
    #plt.plot(real_x, path.velocity_limit, '-x')

    #plt.show()

  
    return










