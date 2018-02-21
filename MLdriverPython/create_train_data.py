from planner import Planner
import numpy as np
import random
import time
from library import *
#from linear_reg_Qlearn import Network

plan = Planner()


path = plan.create_path_in_run(6000*2,"path3.txt")#run vehicle in simulator randomaly and save data in file

#path = plan.read_path_data("path1.txt")
#lenght_max = 70
#path.comp_path_parameters()

#data = split_to_data(path,lenght_max)
##filter data - remove paths with more then 2 velocity steering switches
##choose the best paths compared to the other similar paths 
#save_data(data,"data1.txt")
#print("data saved")
#paths = split_to_paths(path,lenght_max)
#paths = plan.paths_to_local(paths)
#save_paths(paths)



#paths = read_paths()

#train(paths)








