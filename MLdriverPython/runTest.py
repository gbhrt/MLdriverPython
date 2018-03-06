from planner import Planner 
import numpy as np
from library import *
from classes import PathManager
from plot import Plot

if __name__ == "__main__": 
    pm = PathManager()
    plot = Plot()
    #pl.create_path_in_run(0,"path_1.txt")
    path_name = 'splited_files\\random_paths2.txt'
    pl = Planner()
    pl.start_simple()
    pl.load_path(path_name)
    pl.new_episode(points_num = 1000)
    pl.simulator.send_path(pl.desired_path)
    pl.desired_path.comp_path_parameters()
    vel = 5
    pl.desired_path.set_velocity(vel)
    max_index = len(pl.desired_path.position)
    index = pl.find_index_on_path(0)
    while index < max_index-1:#while not reach the end
        target_index = pl.select_target_index(index)
        steer_ang = comp_steer(pl.simulator.vehicle,pl.desired_path.position[target_index])#target in global
        pl.simulator.send_drive_commands(vel,steer_ang) #send commands
        pl.simulator.get_vehicle_data()#read data (respose from simulator to commands)
        index = pl.find_index_on_path(index)#asume index always increase
        pl.update_real_path()
    pl.stop_vehicle()

    pl.end()
    #plot.close()
    plot.plot_path(pl.real_path)
    print("done")

