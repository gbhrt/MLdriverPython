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


commands = [[1.0,0.1]]*5+[[1.0,-0.1]]*5+[[-1.0,0.1]]*5+[[-1.0,-0.1]]*5
pl = planner.Planner(mode = "torque")

step_time = 0.2
N = 10
vec_vehicle_data = []
vec_t = []
for j in range(N):
    pl.simulator.reset_position()
    pl.stop_vehicle()
    time.sleep(1.0)
    pl.restart()#
    last_time = [time.clock()]
    pl.simulator.get_vehicle_data()
    t = [0]
    vehicle_data = [copy.deepcopy(pl.simulator.vehicle)]
    start_time = time.clock()

    for i in range(len(commands)):
        pl.torque_command(commands[i][0],commands[i][1])
        lib.wait_until_end_step(last_time,step_time)
        pl.simulator.get_vehicle_data()
        t.append(time.clock() -start_time)
        vehicle_data.append(copy.deepcopy(pl.simulator.vehicle))
    pl.stop_vehicle()


    vec_vehicle_data.append(vehicle_data)
    vec_t.append(t)


pl.end()

plt.figure(1)
for vehicle_data in vec_vehicle_data:
    x = [vehicle.position[0] - vehicle_data[0].position[0] for vehicle in vehicle_data]
    y = [vehicle.position[1] - vehicle_data[0].position[1] for vehicle in vehicle_data]
    plt.plot(x,y,'.')

plt.figure(2)
for vehicle_data, time1 in zip(vec_vehicle_data,vec_t):
    x = [t for t in time1]
    y = [vehicle.velocity[1]for vehicle in vehicle_data]
    plt.plot(x,y,'.')
plt.figure(3)
for vehicle_data, time1 in zip(vec_vehicle_data,vec_t):
    x = [t for t in time1]
    y = [vehicle.angle[2] for vehicle in vehicle_data]
    plt.plot(x,y,'.')

plt.show()

