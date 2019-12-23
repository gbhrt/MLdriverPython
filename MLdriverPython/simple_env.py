import matplotlib.pyplot as plt
import numpy as np
import time
import copy

import actions
import target_point
import agent
import hyper_parameters
import planner
import library as lib


def test():
    HP = hyper_parameters.ModelBasedHyperParameters()
    Agent = agent.Agent(HP)
    #fig,ax = plt.subplots(1)
    #ax.axis('equal')
    #plt.ion()

    stop_flag = False
    
    for i in range(100):
        t = time.clock()



        #plt.ioff()
        #plt.show()



