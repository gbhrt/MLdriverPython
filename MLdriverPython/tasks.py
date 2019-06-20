import time
import random
from planner import Planner
import numpy as np

from library import *
import classes 
import data_manager
from plot import Plot
import policyBasedNet
import agent_lib as pLib

def create_path_in_run(points,file_name):
    pl = Planner()
    

    j=0
    resolution  = 0.01 #time [sec]
    pl.restart()
        
    time.sleep(1.)
        
    def run_const():
        pl.simulator.get_vehicle_data()#request data from simulator
        #vh = pl.vehicle_to_local()# to_local(np.asarray(vh.position),np.asarray(initState.position),initState.angle[1])
        #pl.real_path.position.append(vh .position)
        #pl.real_path.angle.append(vh.angle)
        #pl.real_path.steering.append(vh.steering)
        #pl.real_path.velocity.append(vh.velocity)
        #time.sleep(resolution)

        pl.real_path.position.append(pl.simulator.vehicle.position)
        pl.real_path.angle.append(pl.simulator.vehicle.angle)
        pl.real_path.steering.append(pl.simulator.vehicle.steering)
        pl.real_path.velocity.append(pl.simulator.vehicle.velocity[1])
        time.sleep(resolution)
        return

    ####create random path####
    vel = 5 #constant speed
    while j < points:
        steer_ang = random.uniform(-0.5,0.5)
        pl.simulator.send_drive_commands(vel,steer_ang)#send commands
        lenght = random.randint(0,500)
        print("steer ang: ",steer_ang," lenght: ", lenght)
        for i in range(lenght):
            run_const()
        j+=i
        
    #steerAng  = 0
    #vel = 5
    #pl.simulator.send_drive_commands(vel,steerAng)#send commands
    #for i in range(50):
    #    run_const()
    #steerAng  = 0.5
    #pl.simulator.send_drive_commands(vel,steerAng)#send commands
    #for i in range(50):
    #    run_const()
    #steerAng  = -0.5
    #pl.simulator.send_drive_commands(vel,steerAng)#send commands
    #for i in range(50):
    #    run_const()
    #pl.simulator.send_drive_commands(0,0)#send commands


    #pl.restart()
    pl.stop_vehicle()
    pl.end()
    pl.save_path(pl.real_path,file_name)
    #with open(file_name, 'w') as f:#append data to the file
    #    for i in range (len(pl.real_path.position)):
    #        f.write("%s \t %s\t %s\t %s\t %s\n" % (pl.real_path.position[i][0],pl.real_path.position[i][1],pl.real_path.angle[i][1],pl.real_path.velocity[i],pl.real_path.steering[i]))

        
    return #pl.real_path





