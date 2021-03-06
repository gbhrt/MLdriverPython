#from planner import Planner
#import matplotlib.pyplot as plt
import numpy as np
import random
import math
import json
import classes
import library as lib
import time
import matplotlib.pyplot as plt
import data_manager1
from hyper_parameters import HyperParameters
import copy
import planner

def comp_const_vel_acc(des_vel,vel):
    kp=1.0
    error =  (des_vel - vel)*kp
    return np.clip(error*0.5,-1,1)

pl = planner.Planner(mode = "torque")
des_vel = 5
steer = 0.3

pl.torque_command(0.2,steer = 0.3)

start = time.time()
t = []
vel = []
wheels_vel =[]
roll = []

for i in range(40):
    pl.simulator.get_vehicle_data()
    #acc = comp_const_vel_acc(des_vel,pl.simulator.vehicle.velocity[1])
    #pl.torque_command(acc,steer = steer)
#    #if i% 10 == 0:
#    #pl.simulator.send_drive_commands(pl.simulator.vehicle.velocity +0.276 ,0)#math.sin(0.2*i)  0.276
    t.append(time.time() -start)
    vel.append(pl.simulator.vehicle.wheels[0].vel_n)
   # wheels_vel.append(pl.simulator.vehicle.wheels_angular_vel)
    roll.append(pl.simulator.vehicle.angle[2])
    time.sleep(0.2)
#pl.torque_command(0.5,steer = 0.1)
#for i in range(30):
#    pl.simulator.get_vehicle_data()

##    #if i% 10 == 0:
##    #pl.simulator.send_drive_commands(pl.simulator.vehicle.velocity +0.276 ,0)#math.sin(0.2*i)  0.276
#    t.append(time.time() -start)
#    vel.append(pl.simulator.vehicle.acceleration)
#    wheels_vel.append(pl.simulator.vehicle.wheels_angular_vel)
#    roll.append(pl.simulator.vehicle.angle[2])
#    time.sleep(0.2)
#pl.simulator.send_drive_commands(-0.2 ,0)#math.sin(0.2*i)  0.276

#for i in range(50):
#    pl.simulator.get_vehicle_data()
#    #if i% 10 == 0:
#    #pl.simulator.send_drive_commands(pl.simulator.vehicle.velocity +0.276 ,0)#math.sin(0.2*i)  0.276
#    t.append(time.time() -start)
#    vel.append(pl.simulator.vehicle.velocity)    
#    time.sleep(0.2)
#pl.simulator.send_drive_commands(0.2 ,0)#math.sin(0.2*i)  0.276



pl.stop_vehicle()
pl.end()
plt.figure(1)
#plt.plot(t,np.array(vel)[:,0])
#plt.plot(t,np.array(vel)[:,1])
#plt.plot(t,np.array(vel)[:,2])
plt.plot(t,vel)

plt.figure(2)
plt.plot(t,roll)

plt.show()
##path = classes.Path()
##path.position = [[0,0.05*i,0] for i in range(10000)]
##lib.comp_velocity_limit_and_velocity(path)
#plt.figure(1)
#plt.plot(t,vel)
##plt.plot(path.analytic_time,path.analytic_velocity)
#plt.figure(2)
#plt.plot(t,np.array(wheels_vel)[:,0])
#plt.plot(t,np.array(wheels_vel)[:,1])
#plt.plot(t,np.array(wheels_vel)[:,2])
#plt.plot(t,np.array(wheels_vel)[:,3])

#plt.show()

#pm = classes.PathManager()
#pm.convert_to_json('paths//circle_path.txt','paths//circle_r7_json.txt')

#path = pm.read_path("paths//straight_path_limit_json.txt")
#random.seed(1234)
#for seed in range(100):
#    print(seed)
#path = classes.Path()
#path.position = lib.create_random_path(9000,0.05,seed = 2 )#,seed = 1244 #seed 6, 1000 points simple path 

#path1 = copy.copy(path)
#lib.comp_velocity_limit_and_velocity(path,skip = 10)
##lib.comp_velocity_limit_and_velocity(path1,skip = 10)
##path.comp_curvature()
##path.comp_angle()
##ang = path.angle
#size = 15
##dang = [-(ang[i+1][1] - ang[i][1])/0.05 for i in range(len(path.angle)-1)]
#plt.figure(1)
#plt.plot(np.array(path.position)[:,0],np.array(path.position)[:,1],label = "Path")
#plt.xlabel('x [m]',fontsize = size)
#plt.ylabel('y [m]',fontsize = size)
#plt.legend()
##plt.show()
#plt.figure(2)

##plt.plot(np.array(path.curvature)*50,'o')
##plt.plot(path.distance,path.analytic_velocity_limit,c =(0,0,0),label = "Velocity limit")
##plt.plot(path.distance,path.analytic_velocity,c='r',label = "Velocity")
#plt.plot(path.analytic_time,path.analytic_velocity_limit,c =(0,0,0),label = "Velocity limit")
#plt.plot(path.analytic_time,path.analytic_velocity,c='r',label = "Velocity")
#plt.xlabel('Time [s]',fontsize = size)
#plt.ylabel('Velocity [m/s]',fontsize = size)
#plt.legend()
##plt.plot(path1.analytic_velocity_limit,'x')
##plt.plot(path1.analytic_velocity,'x')
##plt.plot(path.analytic_acceleration,'o')
##plt.plot(dang)
#plt.show()
