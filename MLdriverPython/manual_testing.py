import numpy as np
import random
import math
import classes
import library as lib
import time
import matplotlib.pyplot as plt
import planner

pl = planner.Planner(mode = "torque")
#pl.torque_command(1.0,steer = 0)
t_vec = []
vel_vec = []
pos_vec = []
targets_vec = []

ang_to_target = random.uniform(-40*3.14/180,40*3.14/180)
dist_to_target = random.uniform(8,20)
target_local = [dist_to_target*math.sin(ang_to_target),dist_to_target*math.cos(ang_to_target)]
pl.simulator.get_vehicle_data()
target_global = pl.to_global(target_local)

for run in range(30):
    start = time.time()
    last_time = [start]
    t = []
    vel = []
    pos = []
    targets = []

    ang_to_target = random.uniform(-40*3.14/180,40*3.14/180)
    dist_to_target = 20+random.uniform(8,20)
    next_target = [dist_to_target*math.sin(ang_to_target),dist_to_target*math.cos(ang_to_target)]

    pl.simulator.get_vehicle_data()
    init_pos,init_ang = pl.simulator.vehicle.position,pl.simulator.vehicle.angle[1]

   # target_global = pl.to_global(target)
    next_target_global = pl.to_global(next_target)
    targets.append(target_global)
    targets.append(next_target_global)

    target_local = pl.to_local(target_global)
    targets_vec.append([target_local,next_target])#save local targets
    #path = classes.Path()
    #path.position.append(target_global)
    #path.position.append([target_global[0]+0.2,target_global[1]+0.2])
   # pl.simulator.send_path(path)
    pl.simulator.send_target_points(targets)#send global targets
    for i in range (100):
        pl.simulator.get_vehicle_data()
        t.append(time.time() -start)
        vel.append(pl.simulator.vehicle.velocity)
        pos.append(lib.to_local(pl.simulator.vehicle.position,init_pos,init_ang))
        while not lib.step_now(last_time,0.2):# and stop != [True]: #wait for the next step (after step_time)
        #print("before wait: ",time.time() - self.last_time[0])
            time.sleep(0.00001)

        target_local = pl.to_local(target_global)
        ang_to_target = math.atan2(target_local[0],target_local[1])
        if abs(ang_to_target) >50*3.14/180:
        #if lib.dist(pl.simulator.vehicle.position[0],pl.simulator.vehicle.position[1],
        #                target_global[0],target_global[1]) < 1.0:
            break
    target_global = list(next_target_global)

    #pl.stop_vehicle()


    t_vec.append(t)
    vel_vec.append(vel)
    pos_vec.append(pos)



pl.end()

for t,vel,pos,targets in zip(t_vec,vel_vec,pos_vec,targets_vec):
    plt.figure(1)
   # plt.subplot(211) 
    plt.plot(t,vel)
    plt.figure(2,figsize=(12,12))
   # plt.subplot(212)
    plt.axis('equal')
    plt.plot(np.array(pos)[:,0],np.array(pos)[:,1])
    print(targets)
    #plt.plot(np.array(targets)[:,0],np.array(targets)[:,1],'o')
    plt.plot([targets[0][0],targets[1][0]],[targets[0][1],targets[1][1]],'o')
    plt.show()