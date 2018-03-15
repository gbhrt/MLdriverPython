from library import *
from classes import *
from communicationLib import Comm
import copy
import random
import time
class SimVehicle:#simulator class - include communication to simulator, vehicle state and world state (recived states).
                 #
                 #also additional objects there like drawed path
    #UDP_IP = None
    #UDP_PORT = None
    #comm = None
    #vehicle = Vehicle()

    def __init__(self):
        self.UDP_IP = "127.0.0.1"
        self.UDP_PORT = 5007
        self.vehicle = Vehicle()
    def set_address(self,IP,PORT):
        self.UDP_IP = IP
        self.UDP_PORT = PORT

    def connect(self):
        self.comm = Comm()
        self.comm.connectToServer(self.UDP_IP,self.UDP_PORT)
        return

    def send_drive_commands(self,velocity,steering):#send drive commands to simulator
        dataType = 1
        self.comm.serialize(dataType)
        self.comm.serialize(velocity)
        self.comm.serialize(steering)
        self.comm.sendData()
        self.comm.readData()
        dataType = self.comm.deserialize(1,int)
        return
    def send_path(self,path_des):
        dataType = 2
        self.comm.serialize(dataType)
        self.comm.serialize(len(path_des.position))
        for i in range(len(path_des.position)):
            self.comm.serialize(path_des.position[i][0])
        for i in range(len(path_des.position)):
            self.comm.serialize(path_des.position[i][1])
            #comm.serialize(path_des.position[i].z)
        self.comm.sendData()
        self.comm.readData()
        dataType = self.comm.deserialize(1,int)
        return
    def read_vehicle_data(self):
        self.comm.readData()
        dataType = self.comm.deserialize(1,int)        
        self.vehicle.position = changeZtoY(self.comm.deserialize(3,float))
        self.vehicle.position[2] = 0.
        self.vehicle.angle = change_to_rad(self.comm.deserialize(3,float))
        if self.vehicle.angle[1] > math.pi:#angle form 0 to 2 pi, convert from -pi to pi
            self.vehicle.angle[1] = -(2*math.pi - self.vehicle.angle[1])
        if self.vehicle.angle[0] > math.pi:#angle form 0 to 2 pi, convert from -pi to pi
            self.vehicle.angle[0] = -(2*math.pi - self.vehicle.angle[0])
        if self.vehicle.angle[2] > math.pi:#angle form 0 to 2 pi, convert from -pi to pi
            self.vehicle.angle[2] = -(2*math.pi - self.vehicle.angle[2])
        self.vehicle.backPosition = changeZtoY(self.comm.deserialize(3,float))
        self.vehicle.velocity = self.comm.deserialize(1,float)
        self.vehicle.steering = self.comm.deserialize(1,float)
        return 
    def get_vehicle_data(self):
        dataType = 0
        self.comm.serialize(dataType)
        self.comm.sendData()
        self.read_vehicle_data()
        return 
    def reset_position(self):
        dataType = 3
        self.comm.serialize(dataType)
        command = 0
        self.comm.serialize(command)
        self.comm.sendData()
        self.comm.readData()
        dataType = self.comm.deserialize(1,int)
        time.sleep(1)
        return



class Planner(PathManager):#planner - get and send data to simulator. input - mission, output - simulator performance 
    def __init__(self):
        super().__init__()
        self.desired_path = Path()
        self.reference_free_path = Path()#a path without reference system (start at (0,0) and angle 0)
        self.in_vehicle_reference_path = Path()# reference_free_path shifted and rotated to vehicle position (vehicle on the path start)
        self.simulator = SimVehicle()      
        self.start_time = 0
        self.index = 0
        self.main_index = 0
        self.max_velocity = 30 #global speed limit
    def init_timer(self):
        self.start_time = time.time()
    def get_time(self):
        self.time = time.time() - self.start_time
        return self.time
    def create_path1(self):
        self.target = [0,5,0]
        self.target = self.to_global(self.target)
        res_dist = 100.# resolution (1/m)
        self.create_path("line",10,1./res_dist)
        self.desired_path.comp_path_parameters()
        self.desired_path.set_velocity(0)
        self.simulator.send_path(self.desired_path)
        print("path sended")

    def stop_vehicle(self):
        self.simulator.send_drive_commands(0,0)
        self.wait_for_stop()
    def wait_for_stop(self):
        self.simulator.get_vehicle_data()
        for _ in range(100):
            self.simulator.get_vehicle_data()
            if abs(self.simulator.vehicle.velocity) < 0.01:
                break
            time.sleep(0.1)
        if abs(self.simulator.vehicle.velocity) > 0.01:
            self.simulator.reset_position()
            self.wait_for_stop()

    def restart(self):
        self.stop_vehicle()
        self.init_timer()
        print("restart\n")
        v = Vehicle()
        self.init_state = copy.copy(self.simulator.vehicle)#save initial state of the vehicle(local reference system)
        

    def start_simple(self):
        self.simulator.connect()
        print("connected")
        self.restart()
        print("started simple scene in simulator")
        return
   
    def end(self):
        self.simulator.comm.end_connection()
        return

    def dist_from_target(self):
        return dist(self.simulator.vehicle.position[0],self.simulator.vehicle.position[1],self.target[0],self.target[1])
       
    def to_local(self, position):#pos given in global reference system, convert it to local reference system (located at init_pos in init_ang)
        #return to_local(position,self.init_state.position,self.init_state.angle[1])#to init
        return to_local(position,self.simulator.vehicle.position,self.simulator.vehicle.angle[1]) #to vehicle

    def to_global(self, position):#pos given in local reference system (located at init_pos in init_ang), convert it to global reference system 
        #return to_global(position,self.init_state.position,self.init_state.angle[1])
        return to_global(position,self.simulator.vehicle.position,self.simulator.vehicle.angle[1])
    def path_tranformation_to_local(self,path):#get a path at any location, transform to vehicle position and angle
        trans_path = Path()
        trans_path.distance = copy.copy(path.distance)
        trans_path.velocity_limit = copy.copy(path.velocity_limit)
        trans_path.velocity = copy.copy(path.velocity)
        path_start = path.position[0]
        path_ang = path.angle[0][1] #- math.pi/2
        for i in range(len(path.position)):#path in reference of the start of the path
            trans_path.position.append(to_local(path.position[i],path_start,path_ang )) 
            trans_path.angle.append([0,path.angle[i][1] - path_ang,0])

        trans_path = self.path_to_global(trans_path)
        return trans_path
    def vehicle_to_local(self):
        vh = Vehicle()
        vh = self.simulator.vehicle
        vh.position = self.to_local(self.simulator.vehicle.position)
        vh.angle[1] = self.simulator.vehicle.angle[1] + self.init_state.angle[1]
        return vh
    def vehicle_to_global(self):
        vh = Vehicle()
        vh = self.simulator.vehicle
        vh.position = self.to_global(self.simulator.vehicle.position)
        vh.angle[1] = self.simulator.vehicle.angle[1] - self.init_state.angle[1]
        return vh

    def path_to_global(self,path):
        global_path = Path()
        global_path.distance = copy.copy(path.distance)
        global_path.velocity_limit = copy.copy(path.velocity_limit)
        global_path.velocity = copy.copy(path.velocity)
        for i in range(len(path.position)):
            global_pos = self.to_global(path.position[i])
            global_path.position.append(global_pos) 
            global_path.angle.append([0.,self.simulator.vehicle.angle[1] - path.angle[i][1],0.])
        return global_path
    def path_to_local(self,path):
        local_path = Path()
        local_path.velocity = copy.copy(path.velocity)
        local_path.velocity_limit = copy.copy(path.velocity_limit)
        for i in range(len(path.position)):
            local_pos = self.to_local(path.position[i])
            local_path.position.append(local_pos) 
            local_path.angle.append([0.,self.simulator.vehicle.angle[1] - path.angle[i][1],0.])
        return local_path

    def create_path(self,type,size,resolution):
        self.desired_path = Path()#initialize the path
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

        if type == "line":
            y= 0
            while y < size - resolution:
                self.desired_path.position.append([0,y,0])
                y+=resolution
            self.desired_path = self.path_to_global(self.desired_path)
        return

    def find_index_on_path(self,start_index):#return closest index to vehicle
        distances = [10000. for _ in range(len(self.desired_path.position))]
        for i in range(start_index, len(self.desired_path.position)):
             distances[i] = (self.desired_path.position[i][0] - self.simulator.vehicle.position[0])**2 + (self.desired_path.position[i][1] - self.simulator.vehicle.position[1])**2
        index_min = np.argmin(distances)
        return index_min
    def get_state_space_index(self):
        start_index = 0
        return self.find_index_on_path(start_index)

    def paths_to_local(self,paths):
        for path in paths:
            self.init_state.position = path.position[0].copy()
            self.init_state.angle = path.angle[0].copy()
            path = self.path_to_local(path)
        return paths
    
    def select_target_index(self,index):
        k = 1.
        min = 2
        max = 6

        if index > len(self.desired_path.distance)-1:
            index = len(self.desired_path.distance) - 1
        forward_distance = k*self.simulator.vehicle.velocity#self.desired_path.velocity[index]
        forward_distance = np.clip(forward_distance,min,max)
        target_index = index
        while (self.desired_path.distance[target_index] - self.desired_path.distance[index]) < forward_distance:
            if target_index >= len(self.desired_path.distance) - 1:
                break
            target_index += 1
        return target_index


    def update_desired_path(self,index,vel_command):
        self.desired_path.velocity[index] = vel_command


    def delta_velocity_command(self, delta_vel):
        des_vel = np.clip(self.simulator.vehicle.velocity + delta_vel,0,self.max_velocity)#assume velocity is updated
        target_index = self.select_target_index(self.index)
        steer_ang1 = comp_steer(self.simulator.vehicle,self.desired_path.position[target_index])#target in global
        self.simulator.send_drive_commands(des_vel,steer_ang1) #send commands
        return
    def load_path(self,path_file_name,random = False, compute_velocity_limit_flag = False):
        if random == False:
            self.reference_free_path = self.read_path(path_file_name)#get a path at any location
            if self.reference_free_path == None:
                return 0
        else:
            path_num,self.reference_free_path = self.get_next_random_path()
            if self.reference_free_path == None:
                return -1
        if random or compute_velocity_limit_flag:
            comp_velocity_limit(self.reference_free_path)
            #self.reference_free_path.velocity_limit[-1] = 0
            
        self.reference_free_path.comp_path_parameters()
        self.reference_free_path.comp_angle()
        return path_num
    def new_episode(self , points_num = 10):
        self.index = 0
        self.main_index = 0
        self.in_vehicle_reference_path = self.path_tranformation_to_local(self.reference_free_path)# transform to vehicle position and angle
        self.desired_path = self.copy_path(self.in_vehicle_reference_path,self.main_index,points_num)#just for the first time
    def get_local_path(self,send_path = True, num_of_points = None):
        #local_path = comp_path(pl,main_path_trans,main_index,num_of_points)#compute local path and global path(inside the planner)
        self.index = self.find_index_on_path(0)
        self.main_index += self.index
        self.desired_path = self.copy_path(self.in_vehicle_reference_path,self.main_index,num_of_points)#choose 100 next points from vehicle position
        if send_path:
            self.simulator.send_path(self.desired_path)
        local_path = self.path_to_local(self.desired_path)#translate path in vehicle reference system
        self.desired_path.comp_path_parameters()
        local_path.distance = copy.copy(self.desired_path.distance)
        return local_path
    def check_end(self,deviation = None,max_deviation = 4,max_roll = 0.2,max_pitch = 0.2, state = None):
        #print("main index", self.main_index, "lenght: ",len(self.in_vehicle_reference_path.position))
        if self.main_index >= len(self.in_vehicle_reference_path.position)-1:#end of the main path
            print("end episode - end of the path")
            return 'path end'
        if state != None and state[0] > 0:
            print("end episode - cross limit curve")
            return 'cross'
        if deviation != None and deviation > max_deviation:
            print("end episode - deviation from path is to big")
            return 'deviate'
        if abs(self.simulator.vehicle.angle[0]) > max_pitch or abs(self.simulator.vehicle.angle[2]) > max_roll:
            print("end episode - roll or pitch to high")
            return 'kipp'
        return 'ok'


