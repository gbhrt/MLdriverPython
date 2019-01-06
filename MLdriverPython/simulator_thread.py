from classes import *
from communicationLib import Comm
import shared
import threading
import time
import copy
class SimVehicle:#simulator class - include communication to simulator, vehicle state and world state (recived states).
                 #
                 #also additional objects there like drawed path

    def __init__(self):
        self.UDP_IP = "127.0.0.1"
        self.UDP_PORT = 5007
        self.vehicle = Vehicle()
        self.connected = False
    def set_address(self,IP,PORT):
        self.UDP_IP = IP
        self.UDP_PORT = PORT

    def connect(self):
        self.comm = Comm()
        self.comm.connectToServer(self.UDP_IP,self.UDP_PORT)
        self.connected = True
        return
    def end_connection(self):
        self.comm.end_connection()
        self.connected = False
        return

    def send_drive_commands(self,velocity,steering):#send drive commands to simulator
        data_type = 1
        #print("vel",velocity)
        self.comm.serialize(data_type)
        self.comm.serialize(velocity)
        self.comm.serialize(steering)
        self.comm.sendData()
        self.comm.readData()
        data_type = self.comm.deserialize(1,int)
        error = 0
        if data_type == -1:
           print("error read data____________________________________________________________________________")
           error = 1 
        return error
    def send_path(self,path_des):
        data_type = 2
        self.comm.serialize(data_type)
        self.comm.serialize(len(path_des.position))
        for i in range(len(path_des.position)):
            self.comm.serialize(path_des.position[i][0])
        for i in range(len(path_des.position)):
            self.comm.serialize(path_des.position[i][1])
            #comm.serialize(path_des.position[i].z)
        self.comm.sendData()
        self.comm.readData()
        data_type = self.comm.deserialize(1,int)
        error = 0
        if data_type == -1:
           print("error read data______________________________________________________________________")
           error = 1 
        return error
    def send_target_points(self,points):
        data_type = 5
        self.comm.serialize(data_type)
        self.comm.serialize(len(points))
        for point in points:
            self.comm.serialize(point[0])
        for point in points:
            self.comm.serialize(point[1])
            #comm.serialize(path_des.position[i].z)
        self.comm.sendData()
        self.comm.readData()
        data_type = self.comm.deserialize(1,int)
        error = 0
        if data_type == -1:
           print("error read data______________________________________________________________________")
           error = 1 
        return error
    def read_vehicle_data(self):
        self.comm.readData()
        data_type = self.comm.deserialize(1,int) 
        error = 0
        if data_type == -1:
            return 1  
                 
        self.vehicle.position = lib.changeZtoY(self.comm.deserialize(3,float))
        self.vehicle.position[2] = 0.
        self.vehicle.angle = lib.change_to_rad(self.comm.deserialize(3,float))
        if self.vehicle.angle[1] > math.pi:#angle form 0 to 2 pi, convert from -pi to pi
            self.vehicle.angle[1] = -(2*math.pi - self.vehicle.angle[1])
        if self.vehicle.angle[0] > math.pi:#angle form 0 to 2 pi, convert from -pi to pi
            self.vehicle.angle[0] = -(2*math.pi - self.vehicle.angle[0])
        if self.vehicle.angle[2] > math.pi:#angle form 0 to 2 pi, convert from -pi to pi
            self.vehicle.angle[2] = -(2*math.pi - self.vehicle.angle[2])
       # self.vehicle.backPosition = lib.changeZtoY(self.comm.deserialize(3,float))
        self.vehicle.velocity = lib.changeZtoY(self.comm.deserialize(3,float))
        self.vehicle.angular_velocity = self.comm.deserialize(3,float)#in radians
        self.vehicle.acceleration = lib.changeZtoY(self.comm.deserialize(3,float))
        self.vehicle.angular_acceleration =self.comm.deserialize(3,float)#in radians
        wheels_angular_vel = lib.change_to_rad(self.comm.deserialize(4,float))
        wheels_vel = self.comm.deserialize(8,float)
        j = 0
        for i in range(4):
            self.vehicle.wheels[i].angular_vel = wheels_angular_vel[i] 
            self.vehicle.wheels[i].vel_n = wheels_vel[j]
            self.vehicle.wheels[i].vel_t = wheels_vel[j+1]
            j+=2
        self.vehicle.steering = self.comm.deserialize(1,float)
        self.vehicle.last_time_stamp = self.comm.deserialize(1,float)
        self.vehicle.input_time = self.comm.deserialize(1,float)

        return error
    def get_vehicle_data(self):
        data_type = 0
        self.comm.serialize(data_type)
        self.comm.sendData()
        error = self.read_vehicle_data()
        return error 
    def reset_position(self):
        data_type = 3
        self.comm.serialize(data_type)
        command = 0
        self.comm.serialize(command)
        self.comm.sendData()
        self.comm.readData()
        data_type = self.comm.deserialize(1,int)
        time.sleep(1)
        error = 0
        if data_type == -1:
            print("error read data__________________________________________________________")
            error = 1
        return error


def copy_vehicle(vehicle2):#copy 2 to 1
        vehicle1 = Vehicle()
        vehicle1 = copy.deepcopy(vehicle2)
        return vehicle1

def copy_path(path2):
    path1 = Path()
    path1 = copy.deepcopy(path2)
    return path1

def communication_loop(simulatorShared):
    print("loop")
    simulator = SimVehicle()   
    simulatorShared.vehicle
    try:
        simulator.connect()
        simulatorShared.connected = True
        print("connected")
    except:
        simulatorShared.connected = False
        print("cannot connect to simulator")

    while not simulatorShared.exit:
        simulator.get_vehicle_data()#update vehicle data continuesly
        with simulatorShared.Lock:
            simulatorShared.vehicle = copy_vehicle(simulator.vehicle)#save last data in the shared resorce
        if simulatorShared.send_commands_flag:
            with simulatorShared.Lock:
                steering = simulatorShared.commands[1]
                acc = simulatorShared.commands[0]
            simulator.send_drive_commands(steering,acc)
            simulatorShared.send_commands_flag = False

        if simulatorShared.send_path_flag:
            with simulatorShared.Lock:
                path = copy_path(simulatorShared.path)
            simulator.send_path(path)
            simulatorShared.send_path_flag = False
        if simulatorShared.reset_position_flag:
            simulator.reset_position()
            simulatorShared.reset_position_flag = False
        if simulatorShared.end_connection_flag:
            simulator.comm.end_connection()
            simulatorShared.connected = False
            simulatorShared.exit = True
        time.sleep(0.01);
            
    
        

        
    

    
class LocalSimulator:#a local instance of the data in the simVehicle class. 
    def __init__(self,simulatorShared):
        self.simulatorShared = simulatorShared
        self.connected = False
        self.vehicle = Vehicle()
    def get_vehicle_data(self):
        with self.simulatorShared.Lock:
            self.vehicle = copy_vehicle(self.simulatorShared.vehicle)

    def send_drive_commands(self,velocity,steering):#send drive commands to simulator
        with self.simulatorShared.Lock:
            self.simulatorShared.commands[0] = steering
            self.simulatorShared.commands[1] = velocity
            self.simulatorShared.send_commands_flag = True

    def send_path(self,path):
        with self.simulatorShared.Lock:
            self.simulatorShared.path = copy_path(path)
            self.simulatorShared.send_path_flag = True
    def reset_position(self):
        self.simulatorShared.reset_position_flag = True
    def end_connection(self):
        self.simulatorShared.end_connection_flag = True

class simulatorThread (threading.Thread):
   def __init__(self,simulatorShared):
        threading.Thread.__init__(self)
        self.simulatorShared = simulatorShared

              
   def run(self):
        print ("Starting " + self.name)
        communication_loop(self.simulatorShared)
        
        print ("Exiting " + self.name)
def start_simulator_connection():#start thread for connection with the simulator 

    simulatorShared = shared.simulatorShared()
    # Create new thread
    simThread = simulatorThread(simulatorShared)
    simThread.start()
    while simulatorShared.connected is None:
        time.sleep(0.1)
    localSim = LocalSimulator(simulatorShared)
    localSim.connected = simulatorShared.connected

    return localSim