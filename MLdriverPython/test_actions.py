import matplotlib.pyplot as plt
import numpy as np

import actions
import agent

def test(Agent):
    targetPoint = actions.TargetPoint()
    S = agent.State()
    S.Vehicle.values = [0,0,0]#v,steer,roll
    S.Vehicle.rel_pos = np.array([0,0])
    S.Vehicle.rel_ang = 0
   # S.Vehicle.abs_pos = np.array([0,0,0])

    targetPoint.rel_pos = np.array([5,3]) #x,y
    targetPoint.vel = 0

    actions.comp_action(Agent.nets,S,Agent.trainHP,targetPoint)


