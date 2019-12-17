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




def one_step_prediction(Agent,replay_memory):
        #one step predictions and plots:
    TransNet_X,TransNet_Y_ = [],[]
    AccNet_X,AccNet_Y_ = [],[]
    SteerNet_X,SteerNet_Y_ = [],[]

    for ind in range(len(replay_memory)-1):
        if replay_memory[ind][3] == True or replay_memory[ind+1][4]: # done flag,time error
            continue
        vehicle_state = replay_memory[ind][0]
        action = replay_memory[ind][2]
        vehicle_state_next = replay_memory[ind+1][0]
        rel_pos = replay_memory[ind+1][1]


        TransNet_X.append(vehicle_state+action)
        TransNet_Y_.append([vehicle_state_next[i] - vehicle_state[i] for i in range(len(vehicle_state_next))] +rel_pos)

        #SteerNet_X.append(vehicle_state+[action[0],vehicle_state_next[vehicle_state_next.vehicle_ind_data['roll']]])
        #SteerNet_Y_.append([action[1]])
    print("legal samples num:",len(TransNet_X))
    

    #print(Agent.nets.SteerNet.evaluate(np.array(SteerNet_X),np.array(SteerNet_Y_)))
    #SteerNet_Y = Agent.nets.SteerNet.predict(np.array(SteerNet_X))
    #plot_comparison(np.array(SteerNet_Y_)[:,0], np.array(SteerNet_Y)[:,0],"steer action")

    print(Agent.nets.TransNet.evaluate(np.array(TransNet_X),np.array(TransNet_Y_)))

    TransNet_Y = Agent.nets.TransNet.predict(np.array(TransNet_X))
    #Direct = direct_method.directModel()
    #TransNet_Y = Direct.predict(TransNet_X)

    vehicle_state_vec,action_vec,vehicle_state_next_vec,rel_pos_vec,vehicle_state_next_pred_vec,rel_pos_pred_vec = [],[],[],[],[],[]
    for x,y_,y in zip(TransNet_X,TransNet_Y_,TransNet_Y):
        vehicle_state_vec.append( x[:len(Agent.trainHP.vehicle_ind_data)])
        action_vec.append(x[len(Agent.trainHP.vehicle_ind_data):])
        vehicle_state_next = y_[:len(Agent.trainHP.vehicle_ind_data)]
        vehicle_state_next_vec.append([vehicle_state_next[i] + vehicle_state_vec[-1][i] for i in range(len(vehicle_state_next))])
        rel_pos_vec.append(y_[len(Agent.trainHP.vehicle_ind_data):])
        vehicle_state_next_pred = y[:len(Agent.trainHP.vehicle_ind_data)]
        vehicle_state_next_pred_vec.append([vehicle_state_next_pred[i] + vehicle_state_vec[-1][i] for i in range(len(vehicle_state_next))])
        rel_pos_pred_vec.append(y[len(Agent.trainHP.vehicle_ind_data):])

    return vehicle_state_next_vec,vehicle_state_next_pred_vec,rel_pos_vec,rel_pos_pred_vec




def get_n_step_states(Agent,replay_memory, n):
    final_state_vec = []
    final_state_vec_pred = []
    final_pos_vec = []
    final_pos_vec_pred = []
    final_ang_vec = []
    final_ang_vec_pred = []
    for i in range(len(replay_memory)-n):
        replay_memory_short = replay_memory[i:i+n]
        vehicle_state_vec,action_vec,abs_pos_vec,abs_ang_vec = real_to_abs_n_steps(replay_memory_short)
        if len(vehicle_state_vec) < n:
            continue
        pred_vehicle_state_vec,pred_abs_pos_vec,pred_abs_ang_vec = predict_n_steps(Agent,vehicle_state_vec[0],abs_pos_vec[0],abs_ang_vec[0],action_vec)

        final_state_vec.append(vehicle_state_vec[-1])
        final_state_vec_pred.append(pred_vehicle_state_vec[-1])

        final_pos_vec.append(abs_pos_vec[-1])
        final_pos_vec_pred.append(pred_abs_pos_vec[-1])

        final_ang_vec.append(abs_ang_vec[-1])
        final_ang_vec_pred.append(pred_abs_ang_vec[-1])

    return final_state_vec,final_state_vec_pred,final_pos_vec,final_pos_vec_pred, final_ang_vec, final_ang_vec_pred
def real_to_abs_n_steps(replay_memory_short):#get a short segment from replay memory. try to compute abs pos from start to end and predict path
    vehicle_state_vec = [replay_memory_short[0][0]]
    action_vec = [replay_memory_short[0][2]]
    abs_ang_vec = [0]
    abs_pos_vec = [[0,0]]#abs relative to the segment begining
    for ind in range(1,len(replay_memory_short)):
        if replay_memory_short[ind][4]:#if time error - break the integration
            break
        vehicle_state_vec.append( replay_memory_short[ind][0])
        action_vec.append( replay_memory_short[ind][2])
         

        rel_pos = replay_memory_short[ind][1][:2]
        rel_ang = replay_memory_short[ind][1][2:][0]

        abs_pos,abs_ang = comp_abs_pos_ang(rel_pos,rel_ang,abs_pos_vec[-1],abs_ang_vec[-1])
        abs_pos_vec.append(abs_pos)
        abs_ang_vec.append(abs_ang)
        if replay_memory_short[ind][3]:#if done or replay_memory_short[ind][4] or replay_memory_short[ind][5]: # done flag,time errors
            break
    return vehicle_state_vec,action_vec,abs_pos_vec,abs_ang_vec

def predict_n_steps(Agent,vehicle_state,abs_pos,abs_ang,action_vec):#
    vehicle_state_vec = [vehicle_state]
    abs_ang_vec = [abs_ang]
    abs_pos_vec = [abs_pos]#abs relative to the segment begining
    for ind in range(len(action_vec)-1):
        X= [vehicle_state_vec[ind]+action_vec[ind]]
        y = Agent.nets.TransNet.predict(np.array(X))[0]

        delta_values = y[:len(Agent.trainHP.vehicle_ind_data)].tolist()
        vehicle_state_vec.append( [vehicle_state_vec[-1][i]+delta_values[i] for i in range(len(delta_values))])

        rel_pos = y[len(Agent.trainHP.vehicle_ind_data):len(Agent.trainHP.vehicle_ind_data)+2]
        rel_ang = y[len(Agent.trainHP.vehicle_ind_data)+2:]

        abs_pos,abs_ang = comp_abs_pos_ang(rel_pos,rel_ang,abs_pos_vec[-1],abs_ang_vec[-1])
        abs_pos_vec.append(abs_pos)
        abs_ang_vec.append(abs_ang)

    return vehicle_state_vec,abs_pos_vec,abs_ang_vec

def get_all_n_step_states(Agent,replay_memory, n):#from 1 step to n steps
    state_vec = []
    state_vec_pred = []
    pos_vec = []
    pos_vec_pred = []
    ang_vec = []
    ang_vec_pred = []

    for i in range(len(replay_memory)-n):
        replay_memory_short = replay_memory[i:i+n]
        vehicle_state_vec,action_vec,abs_pos_vec,abs_ang_vec = real_to_abs_n_steps(replay_memory_short)
        action_vec = action_vec[:len(vehicle_state_vec)]
        pred_vehicle_state_vec,pred_abs_pos_vec,pred_abs_ang_vec = predict_n_steps(Agent,vehicle_state_vec[0],abs_pos_vec[0],abs_ang_vec[0],action_vec)

        state_vec.append(vehicle_state_vec)
        state_vec_pred.append(pred_vehicle_state_vec)

        pos_vec.append(abs_pos_vec)
        pos_vec_pred.append(pred_abs_pos_vec)

        ang_vec.append(abs_ang_vec)
        ang_vec_pred.append(pred_abs_ang_vec)

    all_n_state_vec = []
    all_n_state_vec_pred = []
    all_n_pos_vec = []
    all_n_pos_vec_pred = []
    all_n_ang_vec = []
    all_n_ang_vec_pred = []
    for i in range(1,n):
        n_state_vec = []
        n_state_vec_pred = []
        n_pos_vec = []
        n_pos_vec_pred = []
        n_ang_vec = []
        n_ang_vec_pred = []
        for j in range(len(state_vec)):
            if i >= len(state_vec[j]):
                continue
            n_state_vec.append(state_vec[j][i])
            n_state_vec_pred.append(state_vec_pred[j][i])
            n_pos_vec.append(pos_vec[j][i])
            n_pos_vec_pred.append(pos_vec_pred[j][i])
            n_ang_vec.append(ang_vec[j][i])
            n_ang_vec_pred.append(ang_vec_pred[j][i])
        if len(n_state_vec) == 0:#if no sample reached that lenght, we can break 
            break
        all_n_state_vec.append(n_state_vec)
        all_n_state_vec_pred.append(n_state_vec_pred)
        all_n_pos_vec.append(n_pos_vec)
        all_n_pos_vec_pred.append(n_pos_vec_pred)
        all_n_ang_vec.append(n_ang_vec)
        all_n_ang_vec_pred.append(n_ang_vec_pred)

    

    return all_n_state_vec,all_n_state_vec_pred,all_n_pos_vec,all_n_pos_vec_pred,all_n_ang_vec,all_n_ang_vec_pred

def comp_ac_var(Agent, n_state_vec,n_state_vec_pred,type = "mean_error"):
    var_vec = []
    for final_state_vec,final_state_vec_pred in zip(n_state_vec,n_state_vec_pred):     
        vel_vec = np.array(final_state_vec)[:,Agent.trainHP.vehicle_ind_data["vel_y"]]
        steer_vec = np.array(final_state_vec)[:,Agent.trainHP.vehicle_ind_data["steer"]]
        real = np.array([Agent.Direct.comp_LTR(vel,steer) for vel,steer in zip(vel_vec,steer_vec)])

        vel_vec = np.array(final_state_vec_pred)[:,Agent.trainHP.vehicle_ind_data["vel_y"]]
        steer_vec = np.array(final_state_vec_pred)[:,Agent.trainHP.vehicle_ind_data["steer"]]
        pred = np.array([Agent.Direct.comp_LTR(vel,steer) for vel,steer in zip(vel_vec,steer_vec)])
        error = pred - real
        if type == "mean_error":
            var_vec.append((np.abs(error)).mean())
        else:
            var_vec.append(np.sqrt( np.var(error)))

    return var_vec
            

def comp_var(Agent, n_state_vec,n_state_vec_pred,n_pos_vec,n_pos_vec_pred,n_ang_vec,n_ang_vec_pred,type = "mean_error"):
    var_vec = []
    mean_vec = []
    pos_var_vec = []
    pos_mean_vec = []
    ang_var_vec = []
    ang_mean_vec = []

    
    for final_state_vec,final_state_vec_pred,final_pos_vec,final_pos_vec_pred, final_ang_vec, final_ang_vec_pred in zip(n_state_vec,n_state_vec_pred,n_pos_vec,n_pos_vec_pred,n_ang_vec,n_ang_vec_pred):
    #for n in n_list:
      #  final_state_vec,final_state_vec_pred,final_pos_vec,final_pos_vec_pred, final_ang_vec, final_ang_vec_pred = predict_lib.get_n_step_states(Agent,replay_memory,n)


        

        #print("n",n,"samples num:",len(final_state_vec))
        #compute variance for n
        val_var = []
        val_mean = []
        for feature,ind in Agent.trainHP.vehicle_ind_data.items():
            real = np.array(final_state_vec)[:,ind]
            pred = np.array(final_state_vec_pred)[:,ind]
            error = pred - real
            if type == "mean_error":
                val_var.append((np.abs(error)).mean())
            else:
                val_var.append(np.sqrt( np.var(error)))
            
            val_mean.append(np.mean(error))
        pos_var = []
        pos_mean = []
        for ind in range(2):
            real = np.array(final_pos_vec)[:,ind]
            pred = np.array(final_pos_vec_pred)[:,ind]
            error = pred - real
            if type == "mean_error":
                pos_var.append((np.abs(error)).mean())
            else:
                pos_var.append(np.sqrt( np.var(error)))
            #pos_var.append(np.sqrt( (error**2).mean()))
            pos_mean.append(np.mean(error))

        real = np.array(final_ang_vec)
        final_ang_vec_pred = [item for sublist in final_ang_vec_pred for item in sublist]
        pred = np.array(final_ang_vec_pred)
        
        error = pred - real
        if type == "mean_error":
            ang_var = (np.abs(error)).mean()
        else:
            ang_var = np.sqrt( np.var(error))

        ang_mean = np.mean(error)


        var_vec.append(val_var)
        mean_vec.append(val_mean)
        pos_var_vec.append(pos_var)
        pos_mean_vec.append(pos_mean)
        ang_var_vec.append(ang_var)
        ang_mean_vec.append(ang_mean)

    return var_vec,mean_vec,pos_var_vec,pos_mean_vec,ang_var_vec,ang_mean_vec


