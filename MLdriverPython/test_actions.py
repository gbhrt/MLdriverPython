import matplotlib.pyplot as plt
import numpy as np
import time
import copy

import actions
import target_point
import agent
import hyper_parameters
import planner
import steering_lib
import library as lib


def test():
    HP = hyper_parameters.ModelBasedHyperParameters()
    Agent = agent.Agent(HP)
    targetPoint = target_point.TargetPoint()
    S = agent.State()
    S.Vehicle.values = [0.0,0.0,0.0]#v,steer,roll
    #S.Vehicle.rel_pos = np.array([0.0,0.0])
    #S.Vehicle.rel_ang = 0.0
    #S.Vehicle.abs_pos = np.array([0.0,0.0])
    S.Vehicle.abs_pos = np.array([113.0,26.0])
    S.Vehicle.abs_ang = 0.78

    #targetPoint.abs_pos = np.array([-5.0,7.0]) #x,y
    #targetPoint = actions.target_to_vehicle(targetPoint,S.Vehicle)

    targetPoint.vel = 2.0
    #for _ in range(100):
    #    try:
    #        x = float(input("insert taget x:"))
    #        y = float(input("insert taget y:"))
    #    except:
    #        print("input error")
    #        continue

    #x = 10
    #y = 10
    #targetPoint.abs_pos = np.array([x,y]) #x,y
    #targetPoint = actions.comp_rel_target(targetPoint,S.Vehicle)
    stop_flag = False

    targetPoint.rel_pos = [10,10]
    targetPoint = actions.comp_abs_target(targetPoint,S.Vehicle)  
        

    if actions.plot_states_flag: actions.draw_target(targetPoint)
    StateVehicle_vec = [copy.deepcopy(S.Vehicle)]
    while targetPoint.rel_pos[1] > 0:
        t = time.clock()
        acc,steer,_,_ = actions.comp_action(Agent.nets,S,Agent.trainHP,targetPoint,stop_flag)
        print("comp actions time:",time.clock() - t)
        S.Vehicle = actions.step(S.Vehicle,acc,steer,Agent.nets.TransNet,Agent.trainHP)
        targetPoint = actions.comp_rel_target(targetPoint,S.Vehicle)

        StateVehicle_vec.append(copy.deepcopy(S.Vehicle))

    #actions.plot_state_vec(StateVehicle_vec)
    #plt.show()
    if actions.plot_states_flag:
        plt.ioff()
        plt.show()




def sim_action_test():
    HP = hyper_parameters.ModelBasedHyperParameters()
    Agent = agent.Agent(HP)
    pl = planner.Planner(mode = "torque")
    
    pl.simulator.get_vehicle_data()
    initVehicle = pl.simulator.vehicle
    initState = agent.State()
    
    
    initState.Vehicle.abs_pos = np.array(initVehicle.position[:2]) 
    initState.Vehicle.abs_ang = initVehicle.angle[1]

    start = time.time()
    t = []
    vel = []
    wheels_vel =[]
    roll = []
    steer_vec = []
    x,y=[],[]
    action_steer_vec = []

    targetPoint_vec = []
    rel_pos_vec = [[50,50],[-50,100]]
    for rel_pos in rel_pos_vec:
        targetPoint = target_point.TargetPoint()
        targetPoint.rel_pos = rel_pos
        targetPoint.vel = 0.2
        targetPoint_vec.append(actions.comp_abs_target(targetPoint,initState.Vehicle))

    print("targetPoint_vec:", targetPoint_vec[0].abs_pos,targetPoint_vec[1].abs_pos ) 

    #targetPoint = target_point.TargetPoint()
    #targetPoint.rel_pos = [100,20]
    #targetPoint = actions.comp_abs_target(targetPoint,initState.Vehicle)
    #targetPoint.vel = 0.2

    acc = 1.0
    dir = -1

    for targetPoint in targetPoint_vec:
        actions.plot_target(targetPoint)
    #if actions.plot_states_flag: actions.draw_target(targetPoint)

    last_time = [time.clock()]
    stop_flag = False
    waitFor = lib.waitFor()
    acc,steer = 0,0
    #actions.plot_target(targetPoint)
    #actions.plot_state(initState.Vehicle)
    S = initState
    #for i in range(100):
    StateVehicle_vec = []
    predic_sVehicle = []

    i = 0
    targetPoint = targetPoint_vec[i]
    while targetPoint.rel_pos[1] > 0:
        
        if waitFor.command == [b'1']:
            i+=1
            targetPoint = targetPoint_vec[min(i,len(targetPoint_vec)-1)]
            print("------------------------targetPoint:",targetPoint.abs_pos)
            waitFor.command = []

        if waitFor.stop == [True]:
            stop_flag = True
        print("stop_flag:",stop_flag)
        if stop_flag: break
        time_step_error = lib.wait_until_end_step(last_time,0.2)
        t.append(last_time[0])
        pl.simulator.get_vehicle_data()

        pl.torque_command(acc,steer)

        S.Vehicle.values = [pl.simulator.vehicle.velocity[1],pl.simulator.vehicle.steering,pl.simulator.vehicle.angle[2]]#v,steer,roll
        S.Vehicle.abs_pos = np.array(pl.simulator.vehicle.position[:2])
        S.Vehicle.abs_ang = pl.simulator.vehicle.angle[1]

        targetPoint = actions.comp_rel_target(targetPoint,S.Vehicle)

        StateVehicle_vec.append(copy.deepcopy(S.Vehicle))
        #targetPoint_vec.append(copy.deepcopy(targetPoint))

        acc,steer,sPedVec,_ = actions.comp_action_from_next_step(Agent.nets,copy.deepcopy(S),Agent.trainHP,targetPoint,acc,steer,stop_flag)
        predic_sVehicle.append(copy.deepcopy(sPedVec[0]))

        
        


        #acc,steer = actions.comp_action(Agent.nets,S,Agent.trainHP,targetPoint)
        #if pl.simulator.vehicle.velocity[1] > 20:
        #    acc = -1
        #    dir = -1
        #if pl.simulator.vehicle.velocity[1] < 10:
        #    acc = 1
        #    dir =1
        #steer = actions.get_dsteer_max(Agent.nets.SteerNet,S.Vehicle.values,acc,dir)

        #steer = np.clip(steer,-0.7,0.7)
        
        x.append(S.Vehicle.abs_pos[0])
        y.append(S.Vehicle.abs_pos[1])

        steer_vec.append(pl.simulator.vehicle.steering)
        action_steer_vec.append(steer)
        #t.append(time.time() -start)
        vel.append(pl.simulator.vehicle.velocity[1])#.wheels[0].vel_n
       # wheels_vel.append(pl.simulator.vehicle.wheels_angular_vel)
        roll.append(pl.simulator.vehicle.angle[2])

        

    pl.stop_vehicle()
    pl.end()

    plt.ioff()
    actions.plot_target(targetPoint)
    StateVehicle_vec.pop(0)
    
    actions.plot_state_vec(StateVehicle_vec)
    actions.plot_state_vec(predic_sVehicle)
    for i in range(len(StateVehicle_vec)):
        print("------------i:",i)
        print("real")
        actions.print_stateVehicle(StateVehicle_vec[i])
        print("pred")
        actions.print_stateVehicle(predic_sVehicle[i])

    plt.figure("steer")
    plt.plot(t,steer_vec,'.',label = "steer")
    plt.plot(t,action_steer_vec,'.',label = "action steer")
    plt.legend()

    plt.figure("vel")
    plt.plot(t,vel,'.')

    plt.figure("roll")
    plt.plot(t,roll,'.')

    #plt.figure("pos")
    
    #plt.plot(x,y,'.')
    plt.show()

if __name__ == "__main__":
    #test()
    sim_action_test()