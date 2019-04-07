import actions

class TargetPoint:#target point - [dx,dy, velocity],
    abs_pos = []
    rel_pos = []
    vel = 0

def comp_targetPoint(nets,state,trainHP):# a simple function, a point on the pathe in front of the vehicle, low velocity. for real, an iterative proccess will be used
    n = 

def comp_actions_from_next_step(nets,state,trainHP,acc,steer,stop_flag = False):
    state.Vehicle = actions.step(state.Vehicle,acc,steer,nets.TransNet,trainHP)

    targetPoint = comp_targetPoint(nets,state,trainHP)

    return actions.comp_action(nets,state,trainHP,targetPoint,stop_flag)