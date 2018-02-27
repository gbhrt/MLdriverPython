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



class Planner:#planner - get and send data to simulator. input - mission, output - simulator performance 
    def __init__(self):
        self.desired_path = Path()
        self.real_path = Path()
        self.reference_free_path = Path()#a path without reference system (start at (0,0) and angle 0)
        self.in_vehicle_reference_path = Path()# reference_free_path shifted and rotated to vehicle position (vehicle on the path start)
        self.simulator = SimVehicle()
        target = [0,0,0]
        simulator = SimVehicle()
        init_state = Vehicle()
        local_vehicle = Vehicle()
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
        while abs(self.simulator.vehicle.velocity) > 0.01:
            self.simulator.get_vehicle_data()
    def restart(self):
        self.stop_vehicle()
        self.init_timer()
        self.real_path = Path()
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

    def create_path_in_run(self,points,file_name):
        self.simulator.connect()
        j=0
        resolution  = 0.01
        self.restart()
        
        time.sleep(1.)
        
        def run_const():
            self.simulator.get_vehicle_data()#request data from simulator
            #vh = self.vehicle_to_local()# to_local(np.asarray(vh.position),np.asarray(initState.position),initState.angle[1])
            #self.real_path.position.append(vh .position)
            #self.real_path.angle.append(vh.angle)
            #self.real_path.steering.append(vh.steering)
            #self.real_path.velocity.append(vh.velocity)
            #time.sleep(resolution)

            self.real_path.position.append(self.simulator.vehicle.position)
            self.real_path.angle.append(self.simulator.vehicle.angle)
            self.real_path.steering.append(self.simulator.vehicle.steering)
            self.real_path.velocity.append(self.simulator.vehicle.velocity)
            time.sleep(resolution)
            return

        ####create random path####
        vel = 5 #constant speed
        while j < points:
            steer_ang = random.uniform(-0.7,0.7)
            self.simulator.send_drive_commands(vel,steer_ang)#send commands
            lenght = random.randint(0,500)
            print("steer ang: ",steer_ang," lenght: ", lenght)
            for i in range(lenght):
                run_const()
            j+=i
        
        #steerAng  = 0
        #vel = 5
        #self.simulator.send_drive_commands(vel,steerAng)#send commands
        #for i in range(50):
        #    run_const()
        #steerAng  = 0.5
        #self.simulator.send_drive_commands(vel,steerAng)#send commands
        #for i in range(50):
        #    run_const()
        #steerAng  = -0.5
        #self.simulator.send_drive_commands(vel,steerAng)#send commands
        #for i in range(50):
        #    run_const()
        #self.simulator.send_drive_commands(0,0)#send commands


        #self.restart()
        self.stop_vehicle()
        self.end()
        with open(file_name, 'w') as f:#append data to the file
            for i in range (len(self.real_path.position)):
                f.write("%s \t %s\t %s\t %s\t %s\n" % (self.real_path.position[i][0],self.real_path.position[i][1],self.real_path.angle[i][1],self.real_path.velocity[i],self.real_path.steering[i]))

        
        return self.real_path

    def read_path_data(self,file_name):#, x,y,steer_ang
        path = Path()
        try:
            with open(file_name, 'r') as f:
                data = f.readlines()
                data = [x.strip().split() for x in data]

                results = []
                for x in data:
                    results.append(list(map(float, x)))
                    pos = [float(x[0]),float(x[1])]
                    path.position.append(pos)
                    ang = [0,float(x[2]),0]
                    path.angle.append(ang)
                    path.velocity_limit.append(float(x[3]))
                    path.steering.append(float(x[4]))
                self.desired_path = path
        except:
            print("cannot read file",file_name)
        return path

    def paths_to_local(self,paths):
        for path in paths:
            self.init_state.position = path.position[0].copy()
            self.init_state.angle = path.angle[0].copy()
            path = self.path_to_local(path)
        return paths
    def select_target_index(self,index):
        k = 5.
        min = 3
        max = 20

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

    def run_on_path(self):
        #from neuralNetwork import neuralNetwork
        #nn = neuralNetwork()
        #nn.restore_session()

        self.desired_path.comp_path_parameters()
        vel = 2
        self.desired_path.set_velocity(vel)
        max_index = len(self.desired_path.position)
        index = self.find_index_on_path(0)
        while index < max_index-1:#while not reach the end
            target_index = self.select_target_index(index)
            steer_ang = comp_steer_learn(self.simulator.vehicle,[self.desired_path.position[target_index][0],self.desired_path.position[target_index][1],0])
            #local_target = self.to_local([self.desired_path.position[target_index][0],self.desired_path.position[target_index][1]])
            #print("predict for: ",[-abs(local_target[0]),local_target[1]]," * ",np.sign(-local_target[0]))
            #steer_ang = nn.predict([-abs(local_target[0]),local_target[1]])
            #print("predict steer: ",steer_ang)
            #steer_ang *= np.sign(-local_target[0])
            

           # print("index: ",index," target_index: ",target_index,"\nlocal_target: ",local_target," steer_ang: ", steer_ang)
            self.simulator.send_drive_commands(vel,steer_ang) #send commands
            self.simulator.get_vehicle_data()#read data (respose from simulator to commands)

            index = self.find_index_on_path(index)#asume index always increase

        self.stop_vehicle()
        return

    def update_desired_path(self,index,vel_command):
        self.desired_path.velocity[index] = vel_command

    def update_real_path(self,dist = None, velocity_limit = None):
        if dist != None:
            self.real_path.distance.append(dist)
        self.real_path.position.append(self.simulator.vehicle.position)
        self.real_path.velocity.append(self.simulator.vehicle.velocity)
        self.real_path.velocity_limit.append(velocity_limit)
        self.real_path.time.append(self.get_time())
      
        #print("velocity: ",self.real_path.velocity)
        #print("time: ",self.real_path.time)
        return
    def delta_velocity_command(self, delta_vel):
        des_vel = np.clip(self.simulator.vehicle.velocity + delta_vel,0,self.max_velocity)#assume velocity is updated
        target_index = self.select_target_index(self.index)
        steer_ang1 = comp_steer(self.simulator.vehicle,self.desired_path.position[target_index])#target in global
        self.simulator.send_drive_commands(des_vel,steer_ang1) #send commands
        return
    def load_path(self,path_file_name):
        self.reference_free_path = self.read_path_data(path_file_name)#get a path at any location
        self.reference_free_path.comp_angle()
        return
    def new_episode(self):
        self.index = 0
        self.main_index = 0
        self.in_vehicle_reference_path = self.path_tranformation_to_local(self.reference_free_path)# transform to vehicle position and angle
        self.desired_path = copy_path(self.in_vehicle_reference_path,self.main_index,10)#just for the first time
    def get_local_path(self,num_of_points = None):
        #local_path = comp_path(pl,main_path_trans,main_index,num_of_points)#compute local path and global path(inside the planner)
        self.index = self.find_index_on_path(0)
        self.main_index += self.index
        self.desired_path = copy_path(self.in_vehicle_reference_path,self.main_index,num_of_points)#choose 100 next points from vehicle position
        self.simulator.send_path(self.desired_path)
        local_path = self.path_to_local(self.desired_path)#translate path in vehicle reference system
        self.desired_path.comp_path_parameters()
        local_path.distance = copy.copy(self.desired_path.distance)
        return local_path
    def check_end(self):
        if self.main_index >= len(self.in_vehicle_reference_path.position)-1:#end of the main path
            return True
        else:
            return False


