#import socket
#import time
#import atexit
#import numpy as np
#import copy

#import sys
#import  tkinter
#tkinter._test()
#sys.ps1 = 'SOMETHING'
#import matplotlib
#matplotlib.use('Qt5Agg')
#import matplotlib.pyplot as plt#.pyplot
#from library import *
#from plot import Plot
#import random
#from communicationLib import Comm
#import matplotlib.pyplot as plt
#import os
#from classes import *

import numpy as np

pm = PathManager()
#plot = Plot()
#plt.plot(np.arange(10))
#plt.show()
#path_name = "paths\\‏‏straight_path_limit2.txt" 
#path = pm.read_path_data(path_name)
#name = r'splits\straight_path_limit_splits1'
#pm.save_path(path, name +str(1)+ '.txt')
#path = pm.read_path("paths\‏‏straight_path_limit2.txt")
pm.split_path("paths\\random_path.txt",700,"splited_files1\\random_paths")
#pm.convert_to_json("paths\\‏‏straight_path_limit2.txt","paths\\‏‏straight_path_limit3.txt.txt")
#print(matplotlib.is_interactive())
#plt.ion()
#print(matplotlib.is_interactive())
#plt.plot([1.6, 2.7])
#plt.draw()
#path = pm.read_path("paths\\path.txt")
#plot.plot_path(path,block = False)

#plt.show()
#comp_velocity_limit(path)
#pos = np.array(path.position)

##plt.plot(pos[:,0],pos[:,1])
#plt.plot(np.arange(len(path.velocity_limit)), np.array(path.velocity_limit))

#plt.show()

print("end")

    