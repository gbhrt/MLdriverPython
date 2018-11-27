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



pl = planner.Planner(mode = "torque")
pl.torque_command(0.2,steer = 0.5)

start = time.time()
t = []
vel = []
wheels_vel =[]


for i in range(20):
    pl.simulator.get_vehicle_data()
    t.append(time.time() -start)
    vel.append(pl.simulator.vehicle.velocity)
    wheels_vel.append(pl.simulator.vehicle.wheels_vel)
    time.sleep(0.2)
pl.torque_command(-0.2,steer = 0.5)
for i in range(30):
    pl.simulator.get_vehicle_data()

    t.append(time.time() -start)
    vel.append(pl.simulator.vehicle.velocity)
    wheels_vel.append(pl.simulator.vehicle.wheels_vel)
    time.sleep(0.2)
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

