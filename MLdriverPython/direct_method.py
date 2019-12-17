import math
import numpy as np
import library as lib
class directModel:
    def __init__(self):
        self.steering_vel = 100#rad/sec
        self.wheel_force = 4195*5.0#N
        self.mass = 3200
        self.acceleration = self.wheel_force / self.mass 
        self.dt = 0.2

        self.fc = 1.0
        self.lenght = 3.6
        self.height = 1.4#0.94#0.86
        self.width = 2.08
        self.g = 9.81
        self.ac_max = self.g*self.width*0.5/self.height * 1.0#maximal cetripetal force
        print("ac_max :",self.ac_max )
        self.max_roll = 0.07
        return 

    def check_stability1(self,vehicle_state,factor = 1.0):
        if abs(vehicle_state[2]) > self.max_roll*factor:#0.2:
            return False
        return True
    def check_stability3(self,state_Vehicle,state_env,max_plan_deviation,roll_var,factor = 1.0):
        dev_flag,roll_flag = 0,0
        path = state_env[0]
        index = state_env[1]

        dev_from_path = lib.dist(path.position[index][0],path.position[index][1],state_Vehicle.abs_pos[0],state_Vehicle.abs_pos[1])#absolute deviation from the path
        if dev_from_path > max_plan_deviation:
            dev_flag = 1

        if abs(state_Vehicle.values[2])+roll_var > self.max_roll*factor: #check the current roll 
            roll_flag = math.copysign(1,state_Vehicle.values[2])
        return roll_flag,dev_flag

    def comp_LTR(self,vel,steer):
        if abs(steer) < 0.001:
            radius = 1000
        else:
            radius = math.sqrt((self.lenght*0.5)**2+(self.lenght/math.tan(steer))**2)
        if radius <0.1:
            print("error radius too small")
            ac = 100
        else:
            ac = vel**2/radius

        return ac/self.ac_max

    def check_stability2(self,state_Vehicle,state_env,max_plan_deviation,var = 0.0,factor = 1.0):
        dev_flag,roll_flag = 0,0
        path = state_env[0]
        index = state_env[1]

        dev_from_path = lib.dist(path.position[index][0],path.position[index][1],state_Vehicle.abs_pos[0],state_Vehicle.abs_pos[1])#absolute deviation from the path
        if dev_from_path > max_plan_deviation:
            dev_flag = 1

        #if abs(state_Vehicle.values[2])+roll_var > self.max_roll*factor: #check the current roll 
        #    roll_flag = math.copysign(1,state_Vehicle.values[2])
        LTR = self.comp_LTR(state_Vehicle.values[0],state_Vehicle.values[1])

        if LTR+var > 1.0*factor:
            print("caution - not stable---------------------")
            roll_flag = 1
        #print("vel:",state_Vehicle.values[0],"steer:",state_Vehicle.values[1],"centipetal acceleration:",ac/self.ac_max)
        return roll_flag,dev_flag

   


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

    