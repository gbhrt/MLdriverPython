#import socket
#import time
#import atexit
#import numpy as np
#import copy
from library import *
#import random
#from communicationLib import Comm
#import matplotlib.pyplot as plt
import os
from classes import *
import matplotlib.pyplot as plt
import numpy as np
pm = PathManager()
#path_name = "paths\\‏‏straight_path_limit2.txt" 
#path = pm.read_path_data(path_name)
#name = r'splits\straight_path_limit_splits1'
#pm.save_path(path, name +str(1)+ '.txt')
#path = pm.read_path("paths\‏‏straight_path_limit2.txt")
#pm.split_path("paths\\random_path.txt",1500,"splited_files\\random_paths")
#pm.convert_to_json("paths\\‏‏straight_path_limit2.txt","paths\\‏‏straight_path_limit3.txt.txt")
path = pm.read_path("paths\\path.txt")

#comp_velocity_limit(path)
#pos = np.array(path.position)

##plt.plot(pos[:,0],pos[:,1])
#plt.plot(np.arange(len(path.velocity_limit)), np.array(path.velocity_limit))

#plt.show()

print("end")
#def restore_Q(file,Q):
#    with open(file, 'r') as f:#append data to the file
#        data = f.readlines()
#        data = [x.strip().split() for x in data]
        
#        results = copy.copy(Q)
        
#        for i in range(len(Q)):
#            for j in range(len(Q[0])):
#                results[i][j] = (list(map(float, data[i])))

#        #for x in data:
#        #    results.append(list(map(float, x)))

#        #for i in len(Q):
#        #    Q[i] = 
#    return np.array(results)
#Q =  Q = np.zeros([9,5,3])
#restore_Q("Q2.txt",Q)
#print(Q)
##sess = restore_session()
###input_data = 0,5.,5.0, 0# start_steering, end_pos_x end_pos_y, end_angle, end_steering

##sess.run(y, feed_dict={x: input_data})
###error = compare_to_compute()

#connectToServer = False

#if(connectToServer):
#    comm = Comm()
#    UDP_IP = "127.0.0.1"
#    UDP_PORT = 5007
#    comm.connectToServer(UDP_IP,UDP_PORT)
    



#    vh = getVehicleData(comm)
    
#    data_radius = 50.
#    #angStep = 0.02
#    #resolution = 0.1# time between each sample
#    #create_data_set(comm,data_radius,angStep,resolution)        
    
#    vel = 2
#    steerAng = 0.1
#    x = -5.
#    y = 5.
#    #target = to_global(np.array([x,y,0]),np.array(vh.position),vh.angle[1])
    

#    ##path_des = create_path_in_run(comm)
#    path_des = read_path_from_file("path1.txt")
#    steer_ang_learn = comp_steer_learn_local([x,y])
#    steer_ang = comp_steer_local([x,y])
#    #steer_ang_learn = comp_steer_learn(vh,target)
##    min = test_point(comm,vel,steer_ang_learn,data_radius,x,y)

#    # circle path and draw it in the simulator
    
#    path_des = path_to_global(path_des,vh)
#    #path_des = createPath("circle",20.,vh.position[0],vh.position[1],vh.angle[1])

#    sendPath(comm,path_des)#send path to simulator for drawing on the map
#    comm.readData()
#    dataType = comm.deserialize(1,int)

#    run_on_path(comm, path_des, vh)




#    comm.end_connection()
        

#    #############################end program#####################################
#    #@atexit.register
#    #def goodbye():
#    #    print ("You are now leaving the Python sector.")

#    input()


    