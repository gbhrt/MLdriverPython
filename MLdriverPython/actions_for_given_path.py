import actions
import copy
import library as lib 


def driving_action(state,nets,trainHP,roll_var):
    steer = actions.steer_policy(state.Vehicle,state.env,nets.SteerNet,trainHP)

    #integration on the next n steps:
    acc_to_try =[1.0,-1.0]#[1.0,0.0,-1.0]
    roll_var += trainHP.one_step_var
    for acc_i,try_acc in enumerate(acc_to_try):
        roll_var = trainHP.init_var+trainHP.one_step_var  # initialize for every search
        print("try acc:",try_acc)
                
        tmp_a_vec = [[try_acc,steer]]#this action will operate on the saved state if not fail
        tmp_state_vec = []
        acc_command = try_acc
        steer_command = steer
        S_Vehicle = copy.deepcopy(state.Vehicle)
        #dev_flag,roll_flag = 0,0
        #roll_flag,dev_flag = actions.check_stability(S,roll_var = roll_var,max_plan_roll = max_plan_roll,max_plan_deviation = max_plan_deviation)
        #if dev_flag == 0 and roll_flag == 0:
        for i in range(0,trainHP.rollout_n):#n times 
            if  roll_var <= trainHP.const_var:
                roll_var += trainHP.one_step_var
            S_Vehicle = actions.step(S_Vehicle,acc_command,steer_command,nets.TransNet,trainHP)                   
            tmp_state_vec.append(S_Vehicle)    
            state.env[1] = lib.find_index_on_path(state.env[0],S_Vehicle.abs_pos)         
            steer_command = actions.steer_policy(S_Vehicle,state.env,nets.SteerNet,trainHP)
            acc_command = actions.acc_policy()#always -1
            tmp_a_vec.append([acc_command,steer_command])

            roll_flag,dev_flag = actions.check_stability(S_Vehicle,state.env,roll_var = roll_var,max_plan_roll = trainHP.max_plan_roll,max_plan_deviation = trainHP.max_plan_deviation)
            if roll_flag != 0 or dev_flag != 0:
                #print("break:",roll_flag,dev_flag)
                break
            if S_Vehicle.values[0] < 3.0:#velocity
                break
        if roll_flag == 0 and dev_flag == 0:
            break


        #try_acc = 1.0
    return try_acc,steer,tmp_state_vec,tmp_a_vec

def emergency_cost(state,acc,steer,nets,trainHP,roll_var):
    #S.Vehicle = tmp_state_vec[0]
    S_Vehicle = copy.deepcopy(state.Vehicle)
    acc_command,steer_command = acc,steer
    tmp_a_vec = [[acc,steer]]#this action will operate on the saved state if not fail
    tmp_state_vec = []
                    
    for i in range(0,trainHP.rollout_n):#n times       
        if  roll_var < trainHP.const_var:
            roll_var += trainHP.one_step_var
        S_Vehicle = actions.step(S_Vehicle,acc_command,steer_command,nets.TransNet,trainHP)                   
        tmp_state_vec.append(S_Vehicle) 
        steer_command = actions.emergency_steer_policy(S_Vehicle,state.env,nets.SteerNet,trainHP)
        acc_command = actions.emergency_acc_policy()#always -1
        tmp_a_vec.append([acc_command,steer_command])
        
        state.env[1] = lib.find_index_on_path(state.env[0],S_Vehicle.abs_pos) 
        emergency_roll_flag,emergency_dev_flag = actions.check_stability(S_Vehicle,state.env,roll_var = roll_var,max_plan_roll = trainHP.max_plan_roll,max_plan_deviation = trainHP.max_plan_deviation)
        if emergency_roll_flag != 0 or emergency_dev_flag != 0:
            break      
        if S_Vehicle.values[0] < 2.0:#velocity
            break
                   
    if emergency_roll_flag == 0 and emergency_dev_flag== 0:#regular policy and emergency policy are ok:
        #next_acc = try_acc
        #next_steer = steer
        #print("a:",next_acc,next_steer)
        cost = 0
    else:
        cost = 200
        #next_acc = try_acc
        #next_steer = steer
    return cost, tmp_state_vec,tmp_a_vec


def comp_MB_action(nets,state,acc,steer,trainHP):
    print("___________________new acc compution____________________________")

    state_env =  state.env 
    StateVehicle_vec = [state.Vehicle] 
    actions_vec = [[acc,steer]]
    StateVehicle_emergency_vec= [state.Vehicle]
    actions_emergency_vec = [[acc,steer]]
    
    emergency_action_active = trainHP.emergency_action_flag#initialize to true just if in emergency mode
    #delta_var = 0.002
    #init_var = 0.0#uncertainty of the roll measurment
    #one_step_var = 0.01#roll variance after a single step
    #const_var = 0.05#roll variance at the future states, constant because closed loop control?
    max_plan_roll = trainHP.max_plan_roll
    max_plan_deviation = trainHP.max_plan_deviation
    roll_var = trainHP.init_var
    
    #stability at the initial state:
    if  trainHP.emergency_action_flag:
        roll_flag,dev_flag = actions.check_stability(state.Vehicle,state.env,roll_var = roll_var,max_plan_roll = max_plan_roll,max_plan_deviation = max_plan_deviation)
    else:
        roll_flag,dev_flag = 0,0
    if roll_flag == 0 and dev_flag == 0:     
        #predict the next unavoidable state (actions already done):      
        state.Vehicle = actions.step(state.Vehicle,acc,steer,nets.TransNet,trainHP)
        StateVehicle_vec.append(state.Vehicle)
        StateVehicle_emergency_vec.append(state.Vehicle)
        state.env[1] = lib.find_index_on_path(state.env[0],state.Vehicle.abs_pos)   
         
        roll_var += trainHP.one_step_var
        #stability at the next state (unavoidable state):
        if  trainHP.emergency_action_flag:
            roll_flag,dev_flag = actions.check_stability(state.Vehicle,state.env,roll_var = roll_var,max_plan_roll = max_plan_roll,max_plan_deviation = max_plan_deviation)
        else:
            roll_flag,dev_flag = 0,0
        if roll_flag == 0 and dev_flag == 0:#first step was Ok

            next_acc,next_steer,tmp_state_vec,tmp_a_vec = driving_action(state,nets,trainHP,roll_var)
            StateVehicle_vec += tmp_state_vec
            actions_vec += tmp_a_vec
            if trainHP.emergency_action_flag:
                print("in emergency")
                #check if the emergency policy is safe after applying the action on the first step
                e_cost,tmp_state_vec,tmp_a_vec = emergency_cost(state,next_acc,next_steer,nets,trainHP,roll_var)
                StateVehicle_emergency_vec+=tmp_state_vec
                actions_emergency_vec+=tmp_a_vec

                if e_cost < trainHP.max_cost:
                    emergency_action_active = False          
               
            else:
                emergency_action_active = False







            #if (emergency_roll_flag != 0 or emergency_dev_flag != 0) and (roll_flag != 0 or dev_flag != 0):#both regular and emergency failed:
            #    print("no solution! will fail!!!")

            #if (emergency_roll_flag != 0 or emergency_dev_flag) and (roll_flag == 0 and not dev_flag):
            #    print("solution only for the regular policy - emergency fail")
            #    next_acc = acc_to_try[-1]#can only be the last
            #    next_steer = steer
            #    emergency_action = False

        else:# first step will cause fail
            print("unavoidable fail - after first step")
            #emergency_pred_vec = pred_vec#save just initial state and sirst step
            #emergency_pred_vec_n = []
            #pred_vec_n = []



        #emergency_roll_flag,emergency_dev_flag = True,True#for the case that already failed

        #if (roll_flag != 0 or dev_flag != 0 ):# and (emergency_roll_flag == 0 and not emergency_dev_flag):#if just emergency is OK:
        #    next_steer = predict_lib.emergency_steer_policy()# steering from the next state, send it to the vehicle at the next state
        #    next_acc = predict_lib.emergency_acc_policy()
        #    emergency_action = True
            

    else:#initial state is already unstable
        print("unstable initial state - before first step")

    if emergency_action_active:
        next_steer = actions.emergency_steer_policy(state.Vehicle,state.env,nets.SteerNet,trainHP) #steering from the next state, send it to the vehicle at the next state
        next_acc  = actions.emergency_acc_policy()#always -1
        print("emergency policy is executed!")


    #return next_acc,next_steer,pred_vec,emergency_pred_vec,roll_flag,dev_flag,emergency_action
    return next_acc,next_steer,StateVehicle_vec,actions_vec,StateVehicle_emergency_vec,actions_emergency_vec,emergency_action_active
    #return next_acc,next_steer,planningData,roll_flag,dev_flag