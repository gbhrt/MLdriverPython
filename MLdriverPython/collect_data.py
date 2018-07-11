
import library as lib
from planner import Planner
import numpy as np
import random
import math
import json
import classes
import time
import agent_lib as a_lib
import matplotlib.pyplot as plt
import copy

def collect_random_data(buffer_size,path,name):
    lt = 0#tmp
    buffer = a_lib.Replay(buffer_size)
    pl = Planner(mode = "torque")
    waitFor = lib.waitFor()
    path_num = 10
    path_lenght = 5000
    for i in range(path_num):
        pl.load_path(10000,source = "create_random")
        pl.simulator.get_vehicle_data()
        pl.new_episode()#compute path in current vehicle position
        local_path = pl.get_local_path()

        state={'pos':[0.0,0.0,0.0],'ang':0.0,'rel_pos':[0.0,0.0,0.0],'rel_ang':0.0,'vel':0.0,'steer':0.0}#state of the vehicle. not enoght to compute action
        next_state = copy.copy(state)
        action={'steer':0.0, 'acc':0}

        steer_des = 0
        last_time = [0]
        step_time = 0.2

        pl.simulator.get_vehicle_data()
        state['pos'] = pl.simulator.vehicle.position
        state['ang'] = pl.simulator.vehicle.angle[1]
        state['steer'] = pl.simulator.vehicle.steering #right is positive

    

        while waitFor.stop != [True]:
            if pl.main_index+10 < len(pl.in_vehicle_reference_path.analytic_velocity):
                vel = pl.in_vehicle_reference_path.analytic_velocity[pl.main_index+10]
            else:
                vel = pl.in_vehicle_reference_path.analytic_velocity[-1]
            kp = 100
            #constant velocity:
            error =  (vel - pl.simulator.vehicle.velocity)*kp
            acc = np.clip(error*0.5,-1,1)
            action['steer'] = pl.torque_command(acc)
            action['acc'] = acc

            while (not lib.step_now(last_time,step_time)):# and stop != [True]: #wait for the next step (after step_time)
                time.sleep(0.00001)

            pl.simulator.get_vehicle_data()

            next_state['pos'] = pl.simulator.vehicle.position
            next_state['vel'] = pl.simulator.vehicle.velocity
            next_state['ang'] = pl.simulator.vehicle.angle[1]
            next_state['steer'] = pl.simulator.vehicle.steering #right is positive
            next_state['rel_pos'] = lib.to_local(next_state['pos'],state['pos'],state['ang'])
            next_state['rel_ang'] = next_state['ang'] - state['ang']
            if next_state['rel_ang']  > math.pi: next_state['rel_ang']  -= 2*math.pi
            if next_state['rel_ang']  < -math.pi: next_state['rel_ang']  += 2*math.pi

        
        
            #save data in buffer: (pos,ang,vel,steer,acc,des_steer) - state of the vehicle: pos,ang,vel,steer. actions: acc,des_steer
            # vel,steer,acc,des_steer,     pos, ang
            #state, action, next_state. 
            # 
            t = time.time()
            print (t - lt)
            lt = t
            buffer.add((state['vel'],state['steer'],action['steer'],action['acc'],
                        next_state['rel_pos'][0],next_state['rel_pos'][1],next_state['rel_ang'],next_state['vel'],next_state['steer']))
            #print ("rel_pos",next_state['rel_pos'])

                    
            #buffer.add((pos_rel_to_last[0],pos_rel_to_last[1],ang_rel_to_last,pl.simulator.vehicle.velocity,steer,acc,steer_des))

            local_path = pl.get_local_path(send_path = False,num_of_points = 100)#just for check end
            mode = pl.check_end(deviation = lib.dist(local_path.position[0][0],local_path.position[0][1],0,0))#check if end of the episode
            if mode != 'ok':
                if mode != 'kipp' and mode != 'seen_path_end':
                    pl.stop_vehicle()
                    pl.simulator.reset_position()

                if  mode == 'kipp': #(i % HP.reset_every == 0 and i > 0) or
                    pl.simulator.reset_position()
                    pl.stop_vehicle()
                break

            state = copy.deepcopy(next_state)
     
        pl.stop_vehicle()

    pl.end()

    buffer.save(path,name)

def load_data(buffer_size,path,name):
    buffer = a_lib.Replay(buffer_size)
    buffer.restore(path,name)
    return buffer
if __name__ == "__main__":

    buffer_size = 10000
    name = "data2"
    path = "\\files\\collect_data\\"
    collect_random_data(buffer_size,path,name)
   # buffer = load_data(buffer_size,path,name)

   # #x,y,ang,vel,steer,acc,des_steer = buffer.sample(len(buffer.memory))
   # plt.plot(np.array(buffer.memory)[:,0])
   # plt.plot(np.array(buffer.memory)[:,7])
   ## plt.plot(np.array(buffer.memory)[:,6])
   # plt.show()