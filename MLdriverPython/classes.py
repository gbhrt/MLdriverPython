import math
import json
class SteerData:
    def __init__(self):
        self.start_vel = 0
        self.start_steering = 0
        self.end_pos = 0
        self.end_angle = 0
        self.end_steering = 0
        self.lenght = 0
    start_vel = 0
    start_steering = 0
    end_pos = 0
    end_angle = 0
    end_steering = 0
    lenght = 0

class Vehicle:
    def __init__(self):
        self.position=[0,0,0]
        self.backPosition = [0,0,0]
        self.angle = [0,0,0]
        self.steering = 0
        self.velocity = 0
class Path:
    def __init__(self):
        self.position = []#vector3D()
        self.backPosition = []
        self.angle = []
        self.curvature = []
        self.velocity = [] #real velocity for a real path and planed velocity for a planned path
        self.velocity_limit = []#velocity limit at each point
        self.steering = []
        self.distance = []
        self.time = []
        
    #position = []#vector3D()
    #backPosition = []
    #angle = []#0.
    #curvature =[]# [0.]
    #velocity = []#[0.]
    #steering = []#[0.]
    #distance = []#[0.]
    def dist(self,x1,y1,x2,y2):
        tmp = (x2-x1)**2 + (y2- y1)**2
        if tmp > 0:
            return math.sqrt(tmp)
        else:
            return 0.
    def comp_path_parameters(self):
        self.distance = []
        
        self.distance.append(0.)
        for i in range (1,len(self.position)):#start from 2
            self.distance.append(self.distance[i-1] + self.dist(self.position[i][0],self.position[i][1],self.position[i-1][0],self.position[i-1][1]))
        return
    def comp_angle(self):
        self.angle =[]
        for i in range (0,len(self.position)-1):#start from 0
           self.angle.append([0,-(math.atan2(self.position[i+1][1] - self.position[i][1],self.position[i+1][0] - self.position[i][0]) - math.pi/2),0])
        self.angle.append(self.angle[-1])
    def set_velocity(self, vel):
        self.velocity = []
        for i in range (len(self.position)):
            self.velocity.append(vel)
        return
    #def compute_path_derivatives(self):
    #    for i in range (2,len(self.position)):#start from 2
    #        self.angle[i] = 

#class Saver:
#    def save(self,file_name,data):
#        with open(file_name, 'w') as f:
#            f.write(


   



