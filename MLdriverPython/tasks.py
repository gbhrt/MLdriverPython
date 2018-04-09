import time
import random
from planner import Planner
import numpy as np

from library import *
import classes 
import data_manager
from plot import Plot
import policyBasedNet
import policyBasedLib as pLib

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
        pl.real_path.velocity.append(pl.simulator.vehicle.velocity)
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

def run_test(path_name, learned_velocity = False):
    pm = classes.PathManager()
    dataManager = data_manager.DataManager()
    plot = Plot()
    pl = Planner()

    stop = []
    command = []
    wait_for(stop,command)#wait for "enter" in another thread - then stop = true
    pl.load_path(path_name,compute_velocity_limit_flag = True)
    pl.new_episode(points_num = 1000)
    pl.simulator.send_path(pl.desired_path)
    pl.desired_path.comp_path_parameters()
    if not learned_velocity: #constant velocity
        vel = 5
        pl.desired_path.set_velocity(vel)
    else:
        restore_name = "policy\\model14.ckpt"
        visualized_points = 300
        acc = 1.5
        step_time = 0.2#0.02
        dv = acc * step_time
        action_space =[-dv,dv]
        features_num = 10
        last_time = [0]
        distance_between_points = 1.0
        net = policyBasedNet.Network(features_num,len(action_space)) #for forward only - no training
        net.restore(restore_name)
    max_index = len(pl.desired_path.position)
    index = pl.find_index_on_path(0)
    while  stop != [True]:#while not reach the end
        if step_now(last_time,step_time):#check if make the next step (after step_time)
            pl.simulator.get_vehicle_data()#read data (respose from simulator to commands)
            target_index = pl.select_target_index(index)
            steer_ang = comp_steer(pl.simulator.vehicle,pl.desired_path.position[target_index])#target in global
            if learned_velocity:
                local_path = pl.get_local_path(send_path = True,num_of_points = visualized_points)#num_of_points = visualized_points
                state = pLib.get_state(pl,local_path,features_num,distance_between_points)
                Pi = net.get_Pi(state)
                a,_ = pLib.choose_action(action_space,Pi)#choose action 
                pl.delta_velocity_command(action_space[a])#update velocity (and steering) and send to simulator. index - index on global path (pl.desired_path)
            else:
                pl.simulator.send_drive_commands(vel,steer_ang) #send commands
       
            #index = pl.find_index_on_path(index)#asume index always increase
            dataManager.update_real_path(pl = pl,velocity_limit = local_path.velocity_limit[0])
            dataManager.save_additional_data(pl,features = state,action = a)
            dev = dist(local_path.position[0][0],local_path.position[0][1],0,0)
            mode = pl.check_end(deviation = dev)#check if end of the episode 
            if mode != 'ok':
                break
    pl.stop_vehicle()

    pl.end()
    #plot.close()
    plot.plot_path_with_features(dataManager,distance_between_points)
    print("done")


if __name__ == "__main__": 
    #pm = PathManager()
    #path = Path()
    #path.position.append([0,0,0])
    #path.position.append([1,0,0])



 
#path = pm.read_path('splited_files\\random_paths2.txt')
    #comp_velocity_limit(path)
    #pm.split_path("paths\\random_path2.txt",2000,"splited_files\\random2\\path_")
    #create_path_in_run(50000,"paths\\random_path2.txt")
    #run_test(path_name = 'splited_files\\random_paths4.txt', learned_velocity = True)
    #HP = pLib.HyperParameters()
    #dataManager = data_manager.DataManager(file = HP.save_name+".txt")
    #dataManager.load_data()
    #dataManager.print_data()
    pl = Planner()
    stop = []
    command = []
    wait_for(stop,command)#wait for "enter" in another thread - then stop = true
    #pl.simulator.send_drive_commands(10,0.427)#max
    pl.simulator.send_drive_commands(10,0.2)
    while stop != [True]:
        pl.simulator.get_vehicle_data()
        print("velocity: ",pl.simulator.vehicle.velocity)
    pl.stop_vehicle()

    pl.end()


