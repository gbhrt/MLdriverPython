import matplotlib.pyplot as plt
import numpy as np
import time

import actions
import agent
import hyper_parameters
import planner
import steering_lib


def test():
    HP = hyper_parameters.ModelBasedHyperParameters()
    Agent = agent.Agent(HP)
    targetPoint = actions.TargetPoint()
    S = agent.State()
    S.Vehicle.values = [0,0,0]#v,steer,roll
    S.Vehicle.rel_pos = np.array([0,0])
    S.Vehicle.rel_ang = 0
    S.Vehicle.abs_pos = np.array([0,0])

    targetPoint.abs_pos = np.array([-5,3]) #x,y
    targetPoint.rel_pos = np.array([-5,3])
    targetPoint.vel = 0

    actions.comp_action(Agent.nets,S,Agent.trainHP,targetPoint)





def sim_action_test():
    HP = hyper_parameters.ModelBasedHyperParameters()
    Agent = agent.Agent(HP)
    pl = planner.Planner(mode = "torque")

    #pl.torque_command(0.2,steer = 0.3)

    start = time.time()
    t = []
    vel = []
    wheels_vel =[]
    roll = []
    steer_vec = []
    action_steer_vec = []
    targetPoint = actions.TargetPoint()
    acc = 1.0
    dir = -1
    for i in range(100):
        pl.simulator.get_vehicle_data()
        
        S = agent.State()
        S.Vehicle.values = [pl.simulator.vehicle.velocity[1],pl.simulator.vehicle.steering,pl.simulator.vehicle.angle[2]]#v,steer,roll
        S.Vehicle.rel_pos = np.array([0,0])
        S.Vehicle.rel_ang = 0
        S.Vehicle.abs_pos = np.array([0,0])
        if pl.simulator.vehicle.velocity[1] > 20:
            acc = -1
            #dir = -1
        if pl.simulator.vehicle.velocity[1] < 10:
            acc = 1
            #dir =1

        #actions.comp_action(Agent.nets,S,Agent.trainHP,targetPoint)
        steer = steering_lib.comp_steer(Agent.nets,S.Vehicle,targetPoint,Agent.trainHP,dir)
        print("steer:",steer)
        steer = np.clip(steer,-0.7,0.7)
        pl.torque_command(acc,steer = steer)

        steer_vec.append(pl.simulator.vehicle.steering)
        action_steer_vec.append(steer)
        t.append(time.time() -start)
        vel.append(pl.simulator.vehicle.velocity[1])#.wheels[0].vel_n
       # wheels_vel.append(pl.simulator.vehicle.wheels_angular_vel)
        roll.append(pl.simulator.vehicle.angle[2])
        time.sleep(0.2)

    pl.stop_vehicle()
    pl.end()

    plt.figure("steer")
    plt.plot(t,steer_vec,'.',label = "steer")
    plt.plot(t,action_steer_vec,'.',label = "action steer")
    plt.legend()

    plt.figure("vel")
    plt.plot(t,vel,'.')

    plt.figure("roll")
    plt.plot(t,roll,'.')

    plt.show()

if __name__ == "__main__":
    test()
    #sim_action_test()