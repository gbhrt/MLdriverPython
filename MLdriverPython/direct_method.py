import math
import numpy as np
import library as lib
class directModel:
    def __init__(self,trainHP):
        self.trainHP = trainHP
        self.steering_vel = 100#rad/sec
        self.wheel_force = 4195*5.0#N
        self.mass = 3200
        self.acceleration = self.wheel_force / self.mass 
        self.dt = 0.2

        self.fc = 1.0
        self.lenght = 3.6
        self.lr = self.lenght/2
        self.height = 1.7#0.94#0.86
        self.width = 2.08
        self.g = 9.81
        self.ac_max = self.g*self.width*0.5/self.height * 1.0#maximal cetripetal force
        #print("ac_max :",self.ac_max )
        self.max_roll = 0.07
        return 

    def check_stability1(self,vehicle_state,factor = 1.0):
        if abs(vehicle_state[self.trainHP.vehicle_ind_data["roll"]]) > self.max_roll*factor:#0.2:
            return False
        return True
    def check_stability3(self,state_Vehicle,state_env,max_plan_deviation,roll_var,factor = 1.0):
        dev_flag,roll_flag = 0,0
        path = state_env[0]
        index = state_env[1]

        dev_from_path = lib.dist(path.position[index][0],path.position[index][1],state_Vehicle.abs_pos[0],state_Vehicle.abs_pos[1])#absolute deviation from the path
        if dev_from_path > max_plan_deviation:
            dev_flag = 1

        if abs(state_Vehicle.values[self.trainHP.vehicle_ind_data["roll"]])+roll_var > self.max_roll*factor: #check the current roll 
            roll_flag = math.copysign(1,state_Vehicle.values[self.trainHP.vehicle_ind_data["roll"]])
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
        LTR = self.comp_LTR(state_Vehicle.values[self.trainHP.vehicle_ind_data["vel_y"]],state_Vehicle.values[self.trainHP.vehicle_ind_data["steer"]])

        if LTR+var > 1.0*factor:
            #print("caution - not stable---------------------")
            roll_flag = 1
        #print("vel:",state_Vehicle.values[0],"steer:",state_Vehicle.values[1],"centipetal acceleration:",ac/self.ac_max)
        return roll_flag,dev_flag

   
    def comp_radius(self,vehicle_state):
        if abs(vehicle_state[self.trainHP.vehicle_ind_data["steer"]]) < 0.001:
            radius = 1000
        else:
            radius = math.sqrt((self.lr)**2+(self.lenght/math.tan(vehicle_state[self.trainHP.vehicle_ind_data["steer"]]))**2)
        return radius

    def check_stability(self,vehicle_state,factor = 1.0):#action

        radius = self.comp_radius(vehicle_state) 
        #print("steer3: ",vehicle_state[1],"radius: ",radius)
        if radius <0.1:
            print("error radius too small")
            ac = 100
        else:
            ac = vehicle_state[0]**2/radius
        print("vel:",vehicle_state[self.trainHP.vehicle_ind_data["vel_y"]],"steer:",vehicle_state[self.trainHP.vehicle_ind_data["steer"]],"centipetal acceleration:",ac/self.ac_max)
        print("factor:",factor)
        if ac > self.ac_max*factor:
            print("caution - not stable---------------------")
            return False
        return True

    def predict_one(self,vehicle_state,action):#vehicle_state - vel,steer,roll. action - acc,steer
        next_vehicle_state = [0]*len(self.trainHP.vehicle_ind_data)
        next_vehicle_state[self.trainHP.vehicle_ind_data["vel_y"]] = action[0]*self.acceleration*self.dt#max(0,)#v+a*dt
        dsteer = action[1] - vehicle_state[self.trainHP.vehicle_ind_data["steer"]]

        next_vehicle_state[self.trainHP.vehicle_ind_data["steer"]] = math.copysign( min(self.steering_vel*self.dt,abs(dsteer)),dsteer)

        #next_vehicle_state[1] = action[1] - vehicle_state[1]
        next_vehicle_state[self.trainHP.vehicle_ind_data["roll"]] = 0
        next_vehicle_state[self.trainHP.vehicle_ind_data["vel_x"]] = 0

        Vy = vehicle_state[self.trainHP.vehicle_ind_data["vel_y"]]
        Vx = vehicle_state[self.trainHP.vehicle_ind_data["vel_x"]]
        #Vy2 = next_vehicle_state[self.trainHP.vehicle_ind_data["vel_y"]]
        slip_ang1 = np.arctan(Vx/Vy)
        slip_ang = abs(np.arctan(self.lr/self.lenght*np.tan(vehicle_state[self.trainHP.vehicle_ind_data["steer"]])))
        #print("slip_ang:",slip_ang,"slip_ang1:",slip_ang1)
        V = Vy#np.sqrt(Vy**2 + Vx**2)
        R1 = self.comp_radius(vehicle_state) 
        R = abs(V/vehicle_state[self.trainHP.vehicle_ind_data["angular_vel_z"]])
        #print("R:",R,"R1:",R1,"V:",V,"steer",vehicle_state[self.trainHP.vehicle_ind_data["steer"]],"omega:",vehicle_state[self.trainHP.vehicle_ind_data["angular_vel_z"]])
        #print("omega_computed:",Vy/R,"omega_measured:",vehicle_state[self.trainHP.vehicle_ind_data["angular_vel_z"]])
        #dang1 = V*self.dt/R
        dang = abs(vehicle_state[self.trainHP.vehicle_ind_data["angular_vel_z"]]*self.dt)
        dx1 = R*(1- np.cos(dang))
        dy1 = R*np.sin(dang)
        #dx1 = Vx*self.dt
        #dy1 = Vy*self.dt

        pos = lib.rotateVec([dx1,dy1],slip_ang)
        dx0,dy0 = pos[0],pos[1]
        #print("V:",V,"R:",R,"dang:",dang,"dx:",dx,"dy:",dy,"steer:",vehicle_state[1],"math.copysign( dx,vehicle_state[1]):",math.copysign( dx,vehicle_state[1]))
        rel_pos,rel_ang = [-math.copysign( dx0,vehicle_state[self.trainHP.vehicle_ind_data["steer"]]),dy0],-math.copysign(dang,vehicle_state[self.trainHP.vehicle_ind_data["steer"]])
        return next_vehicle_state,rel_pos,rel_ang

    def predict(self,X):
        Y = []
        for x in X:
            next_vehicle_state,rel_pos,rel_ang = self.predict_one(x[:len(self.trainHP.vehicle_ind_data)],x[len(self.trainHP.vehicle_ind_data):])
            Y.append(next_vehicle_state+rel_pos+[rel_ang])
        return np.array(Y)
    def evaluate(self,x,y):
        return

    