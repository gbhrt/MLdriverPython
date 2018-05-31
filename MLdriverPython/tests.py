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
import numpy as np
import data_manager1
from hyper_parameters import HyperParameters
import copy

#HP = HyperParameters()
#dm = data_manager1.DataManager(HP.save_file_path,HP.restore_file_path,HP.restore_flag)
#dm.plot_all()


#pl = Planner(mode = "torque")

##pl.simulator.send_drive_commands(30,0)
#start = time.time()
#t = []
#vel = []
#pl.simulator.send_drive_commands(0.2 ,0)#math.sin(0.2*i)  0.276

#for i in range(50):
#    pl.simulator.get_vehicle_data()
#    #if i% 10 == 0:
#    #pl.simulator.send_drive_commands(pl.simulator.vehicle.velocity +0.276 ,0)#math.sin(0.2*i)  0.276
#    t.append(time.time() -start)
#    vel.append(pl.simulator.vehicle.velocity)
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



#pl.stop_vehicle()
#pl.end()
#plt.plot(t,vel)
#plt.show()
#pm = classes.PathManager()
##pm.convert_to_json('paths//straight_path_limit.txt','paths//straight_path_limit_json.txt')

#path = pm.read_path("paths//straight_path_limit_json.txt")
#random.seed(1234)
path = classes.Path()
path.position = lib.create_random_path(500,0.05,seed = 1234)

path1 = copy.copy(path)
lib.comp_velocity_limit_and_velocity(path,skip = 20,init_vel = 25)
#lib.comp_velocity_limit_and_velocity(path1,skip = 10)
#path.comp_curvature()
#path.comp_angle()
#ang = path.angle

#dang = [-(ang[i+1][1] - ang[i][1])/0.05 for i in range(len(path.angle)-1)]
#plt.figure(1)
#plt.plot(np.array(path.position)[:,0],np.array(path.position)[:,1])

#plt.show()
#plt.figure(2)
#plt.plot(np.array(path.curvature)*50,'o')
plt.plot(path.analytic_velocity_limit,'o')
plt.plot(path.analytic_velocity,'o')

#plt.plot(path1.analytic_velocity_limit,'x')
#plt.plot(path1.analytic_velocity,'x')
#plt.plot(path.analytic_acceleration,'o')
#plt.plot(dang)
plt.show()
