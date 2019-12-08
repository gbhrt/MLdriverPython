import math
import numpy as np
class directModel:
    def __init__(self):
        self.steering_vel = 100#rad/sec
        self.wheel_force = 4195*5.0#N
        self.mass = 3200
        self.acceleration = self.wheel_force / self.mass 
        self.dt = 0.2

        self.fc = 1.0
        self.lenght = 3.6
        self.height = 1.7#0.94#0.86
        self.width = 2.08
        self.g = 9.81
        self.ac_max = self.g*self.width*0.5/self.height * 2.5#maximal cetripetal force
        print("ac_max :",self.ac_max )
        return 

    def check_stability1(self,vehicle_state,factor = 1.0):
        if abs(vehicle_state[2]) > 0.2:
            return False
        return True


    def check_stability(self,vehicle_state,factor = 1.0):#action
        if abs(vehicle_state[1]) < 0.001:
            radius = 1000
        else:
            radius = math.sqrt((self.lenght*0.5)**2+(self.lenght/math.tan(vehicle_state[1]))**2)

        #print("steer3: ",vehicle_state[1],"radius: ",radius)
        if radius <0.1:
            print("error radius too small")
            ac = 100
        else:
            ac = vehicle_state[0]**2/radius
        print("vel:",vehicle_state[0],"steer:",vehicle_state[1],"centipetal acceleration:",ac/self.ac_max)
        print("factor:",factor)
        if ac > self.ac_max*factor:
            print("caution - not stable---------------------")
            return False
        return True

    def predict_one(self,vehicle_state,action):#vehicle_state - vel,steer,roll. action - acc,steer
        next_vehicle_state = [0,0,0]
        next_vehicle_state[0] = action[0]*self.acceleration*self.dt#max(0,)#v+a*dt
        dsteer = action[1] - vehicle_state[1]
        if dsteer > 0:
            next_vehicle_state[1] = + min(self.steering_vel*self.dt,dsteer)
        else:
            next_vehicle_state[1] = - min(self.steering_vel*self.dt,-dsteer)
        #next_vehicle_state[1] = action[1] - vehicle_state[1]
        next_vehicle_state[2] = 0
        rel_pos,rel_ang = [0,0],0
        return next_vehicle_state,rel_pos,rel_ang

    def predict(self,X):
        Y = []
        for x in X:
            next_vehicle_state,rel_pos,rel_ang = self.predict_one(x[:3],x[3:])
            Y.append(next_vehicle_state+rel_pos+[rel_ang])
        return Y

    