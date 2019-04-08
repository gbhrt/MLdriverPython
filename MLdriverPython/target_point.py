import actions
import classes
class TargetPoint:#target point - [dx,dy, velocity],
    abs_pos = []
    rel_pos = []
    vel = 0

def comp_targetPoint(nets,state,trainHP):# a simple function, a point on the path in front of the vehicle, low velocity. for real, an iterative proccess will be used
    n = 200
    targetPoint = TargetPoint()
    path = state.env
    targetPoint.abs_pos = path.position[n][:2]
    targetPoint.rel_pos = list(targetPoint.abs_pos)
    targetPoint.vel = 0.5
    return targetPoint

def comp_actions_from_next_step(nets,state,trainHP,acc,steer,stop_flag = False):
    state.Vehicle = actions.step(state.Vehicle,acc,steer,nets.TransNet,trainHP)

    targetPoint = comp_targetPoint(nets,state,trainHP)

    return actions.comp_action(nets,state,trainHP,targetPoint,stop_flag)