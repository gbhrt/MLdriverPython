import copy
import numpy as np
import library as lib
from math import copysign

def steer_policy(abs_pos,abs_ang,path,index,vel):
    return lib.comp_steer_general(path,index,abs_pos,abs_ang,vel)
def emergency_steer_policy():
    return 0.0
def acc_policy():
    return -1.0
def emergency_acc_policy():
    return -1.0

def comp_abs_pos_ang(rel_pos,rel_ang,abs_pos,abs_ang):
    next_abs_pos = lib.to_global(rel_pos,abs_pos,abs_ang)
    next_rel_ang = abs_ang + rel_ang
    return next_abs_pos,next_rel_ang

def check_stability(env,path,index,abs_pos,roll,roll_var = 0.0,max_plan_roll = None,max_plan_deviation = None):
    dev_flag,roll_flag = False,0
    dev_from_path = lib.dist(path.position[index][0],path.position[index][1],abs_pos[0],abs_pos[1])#absolute deviation from the path
    if abs(env.denormalize(roll,'roll'))+roll_var > max_plan_roll: #check the current roll 
        roll_flag = copysign(1,roll)
    if dev_from_path > max_plan_deviation:
        dev_flag = True
        #max_plan_deviation = 10
    return roll_flag,dev_flag
#input: init state and commands 
#output: x dict of the initial state+actions and abs_pos,abs_ang
def initilize_prediction(env,init_state,acc_command,steer_command):
    abs_pos = [0,0]#relative to the local path
    abs_ang = 0
    X = env.create_X([init_state],[[acc_command ,steer_command]])[0]# first action is already executed and known
    return env.X_to_X_dict(X),abs_pos,abs_ang

#input: X dict
#output: X dict after one step
def predict_one_step(net,env,X_dict,abs_pos,abs_ang):
    X = env.dict_X_to_X(X_dict)
    #Y = list(net.get_Y([X])[0])#get Y list from X list
    Y = list(net.predict(np.array([X]))[0])
    Y_dict = env.Y_to_Y_dict(Y)
    for name in env.copy_Y_to_X_names:
        X_dict[name] += Y_dict[name]
        rel_pos = [env.denormalize(Y_dict["rel_pos_x"],"rel_pos_x"),env.denormalize(Y_dict["rel_pos_y"],"rel_pos_y")]
    abs_pos,abs_ang = comp_abs_pos_ang(rel_pos,env.denormalize(Y_dict["rel_ang"],"rel_ang"),abs_pos,abs_ang)
    #pred = [abs_pos,abs_ang,vel,roll,roll_var]
    return X_dict,abs_pos,abs_ang
#input: 
#output:
def predict_n_next1(n,net,env,X_dict,abs_pos,abs_ang,path,emergency_flag = False,max_plan_roll = None,max_plan_deviation = None,roll_var = 0,delta_var = 0.0):
    dev_flag,roll_flag = False,0
    pred_vec = []
    roll_flag,dev_flag = check_stability(env,path,0,abs_pos,X_dict['roll'],roll_var = roll_var,max_plan_roll = max_plan_roll,max_plan_deviation = max_plan_deviation)
    if dev_flag == False and roll_flag == 0:
        for i in range(0,n):#n times       
            X_dict,abs_pos,abs_ang = predict_one_step(net,env,X_dict,abs_pos,abs_ang)
            #roll_var+=delta_var
            #print("roll_var:",roll_var)
            vel = env.denormalize(X_dict['vel_y'],"vel_y")
            index = lib.find_index_on_path(path,abs_pos)
           
            if emergency_flag:
                steer_command = emergency_steer_policy()
                acc_command = emergency_acc_policy()
            else:
                steer_command = steer_policy(abs_pos,abs_ang,path,index,vel)
                acc_command = acc_policy()#always -1


            X_dict["steer_action"] = env.normalize(steer_command,"steer_action")
            X_dict["acc_action"] = acc_command

            roll = env.denormalize(X_dict["roll"],"roll")

            pred_vec.append([abs_pos,abs_ang,vel,roll,roll_var,acc_command,steer_command])

            if vel < 2.0:
                break
            roll_flag,dev_flag = check_stability(env,path,index,abs_pos,X_dict['roll'],roll_var = roll_var,max_plan_roll = max_plan_roll,max_plan_deviation = max_plan_deviation)
            
            if roll_flag != 0 or dev_flag:
                break
            


            
    #print("end loop time:",time.clock() - env.lt)
    #print("pred:", pred_vec)
    #print("roll:", np.array(pred_vec)[:,3])
    #print("vel:", np.array(pred_vec)[:,2])
   
    #print("end--------------------------")
    return pred_vec,roll_flag,dev_flag

#def predict_n_next(n,net,env,init_state,acc_command,steer_command,acc_try = 1.0,emergency_flag = False,max_plan_roll = None,max_plan_deviation = None):
#    if max_plan_roll is None:
#        max_plan_roll = env.max_plan_roll
#    if max_plan_deviation is None:
#        max_plan_deviation = env.max_plan_deviation

#    X_dict,abs_pos,abs_ang = initilize_prediction(env,init_state,acc_command,steer_command)
#    pred_vec = [[abs_pos,abs_ang,init_state['vel_y'],init_state['roll'],0]]
#    pred_vec_n,roll_flag,dev_flag = predict_n_next1(n,net,env,X_dict,abs_pos,abs_ang,init_state['path'],acc_try = 1.0,emergency_flag = emergency_flag,max_plan_roll = max_plan_roll,max_plan_deviation = max_plan_deviation,roll_var = roll_var,delta_var = 0.0)
#    pred_vec+=pred_vec_n
#    return pred_vec,roll_flag,dev_flag

def predict_n_next(n,net,env,init_state,acc_command,steer_command,acc_try = 1.0,emergency_flag = False,max_plan_roll = None,max_plan_deviation = None):
    if max_plan_roll is None:
        max_plan_roll = env.max_plan_roll
    if max_plan_deviation is None:
        max_plan_deviation = env.max_plan_deviation

    abs_pos = [0,0]#relative to the local path
    abs_ang = 0
    
    if emergency_flag:
        steer_command = emergency_steer_policy()
    else:
        steer_command = steer_policy(abs_pos,abs_ang,init_state['path'],0,init_state['vel_y'])
    #print("after steer_command time:",time.clock() - env.lt)
    #print("current action: ",action, "try action: ",acc_try)
 
    dev_flag,roll_flag = False,0
    dev_from_path = lib.dist(init_state['path'].position[0][0],init_state['path'].position[0][1],0,0)#absolute deviation from the path
    print("deviation:", dev_from_path)
    delta_var = 0.0#0.002
    roll_var = max_plan_roll*delta_var
    if abs(init_state['roll'])+roll_var > max_plan_roll: #check the current roll 
        roll_flag = copysign(1,init_state['roll'])
    if dev_from_path > max_plan_deviation:
        dev_flag = True
        max_plan_deviation = 10
    pred_vec = [[abs_pos,abs_ang,init_state['vel_y'],init_state['roll'],roll_var]]#,abs_ang,init_state['vel'],init_state['steer'],init_state['roll']]]
    X = env.create_X([init_state],[[acc_command ,steer_command]])[0]# first action is already executed and known

    X_dict = env.X_to_X_dict(X)
    
    
    if dev_flag == False and roll_flag == 0:
        for i in range(1,n):#n times       
            #print("before prediction time:",time.clock() - env.lt)
            X = env.dict_X_to_X(X_dict)
            
            #print("before get Y time:",time.clock() - env.lt)
            Y = list(net.get_Y([X])[0])#get Y list from X list
            #print("X:",X,"Y:",Y)
            #print("after get Y time:",time.clock() - env.lt)
            Y_dict = env.Y_to_Y_dict(Y)
            #print("after prediction time:",time.clock() - env.lt)
            for name in env.copy_Y_to_X_names:
                Y_dict[name] += X_dict[name]
                X_dict[name] = Y_dict[name]#env.normalize(Y_dict[name],name)

            roll_var+=delta_var
            #print("roll_var:",roll_var)
            #print("X_dict:",X_dict,"Y_dict:",Y_dict)
            X = copy.copy(Y[:len(Y) - 5])#copy the whole relative information (exclude commands, rel pos (2) and rel ang(1))

            abs_pos,abs_ang = comp_abs_pos_ang(Y[-3:-1],Y[-1],abs_pos,abs_ang)#rel_pos = Y[0:2] rel_ang = Y[2] roll Y[5]
            rel_pos = [env.denormalize(Y_dict["rel_pos_x"],"rel_pos_x"),env.denormalize(Y_dict["rel_pos_y"],"rel_pos_y")]
            abs_pos,abs_ang = comp_abs_pos_ang(rel_pos,env.denormalize(Y_dict["rel_ang"],"rel_ang"),abs_pos,abs_ang)#rel_pos = Y[0:2] rel_ang = Y[2] roll Y[5]

            vel = env.denormalize(Y_dict['vel_y'],"vel_y")
            index = lib.find_index_on_path(init_state['path'],abs_pos)
            steer_command =  lib.comp_steer_general(init_state['path'],index,abs_pos,abs_ang,vel)#action steerinit_state['vel_y']

            
            if emergency_flag:
                steer_command = emergency_steer_policy()
                acc_command = emergency_acc_policy()
            else:
                steer_command = steer_policy(abs_pos,abs_ang,init_state['path'],index,vel)
                #acc_command = acc_policy(i,acc_try)
                if i == 1:
                    acc_command = acc_try
                else:
                    acc_command = -1

            X_dict["steer_action"] = env.normalize(steer_command,"steer_action")


            X_dict["acc_action"] = acc_command

            roll = env.denormalize(Y_dict["roll"],"roll")

            pred_vec.append([abs_pos,abs_ang,vel,roll,roll_var])
            if vel < 2.0:
                break

            dev_from_path = lib.dist(init_state['path'].position[index][0],init_state['path'].position[index][1],abs_pos[0],abs_pos[1])#absolute deviation from the path
            #print("deviation:", dev_from_path)

            
            if abs(roll)+roll_var > max_plan_roll: 
                print("fail rool or dev")
                roll_flag = copysign(1,roll)
                break
            if dev_from_path > max_plan_deviation:
                dev_flag = True
                break

            
    #print("end loop time:",time.clock() - env.lt)
    #print("pred:", pred_vec)
    #print("roll:", np.array(pred_vec)[:,3])
    #print("vel:", np.array(pred_vec)[:,2])
   
    print("end--------------------------")
    return pred_vec,roll_flag,dev_flag
