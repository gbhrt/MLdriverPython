import actions
import copy
import library as lib 
import agent


def driving_action(state,nets,trainHP,Direct = None,planningState = None):
    steer = actions.steer_policy(state.Vehicle,state.env,trainHP)#nets.SteerNet

    #integration on the next n steps:
    acc_to_try =[1.0,-1.0]#[1.0,0.0,-1.0]

    for acc_i,try_acc in enumerate(acc_to_try):
        #roll_var = planningState.var[1] #trainHP.init_var+trainHP.one_step_var  # initialize for every search
        #print("try acc:",try_acc)
                
        tmp_a_vec = [[try_acc,steer]]#this action will operate on the saved state if not fail
        tmp_state_vec = []
        acc_command = try_acc
        steer_command = steer
        S_Vehicle = copy.deepcopy(state.Vehicle)
        #dev_flag,roll_flag = 0,0
        #roll_flag,dev_flag = actions.check_stability(S,roll_var = roll_var,max_plan_roll = max_plan_roll,max_plan_deviation = max_plan_deviation)
        #if dev_flag == 0 and roll_flag == 0:
        for i in range(0,trainHP.rollout_n):#n times 
            #if  roll_var <= trainHP.const_var:
            #    roll_var += trainHP.one_step_var
            
            S_Vehicle = actions.step(S_Vehicle,acc_command,steer_command,nets.TransNet if not trainHP.direct_predict_active else Direct,trainHP)                   
            tmp_state_vec.append(S_Vehicle)    
            state.env[1] = lib.find_index_on_path(state.env[0],S_Vehicle.abs_pos)   
    
            steer_command = actions.steer_policy(S_Vehicle,state.env,trainHP)#,nets.SteerNet
            
            acc_command = actions.acc_policy()#always -1
            tmp_a_vec.append([acc_command,steer_command])
            if not trainHP.direct_constrain:  
                roll_flag,dev_flag = actions.check_stability(S_Vehicle,state.env,trainHP,roll_var = planningState.var[i+2],max_plan_roll = trainHP.max_plan_roll,max_plan_deviation = trainHP.max_plan_deviation)
            else:
                roll_flag,dev_flag = Direct.check_stability2(S_Vehicle,state.env,trainHP.max_plan_deviation,planningState.var[i+2],0.5 if planningState.last_emergency_action_active else 1.0)

            if roll_flag != 0 or dev_flag != 0:
                #print("break:",roll_flag,dev_flag)
                break
            if S_Vehicle.values[trainHP.vehicle_ind_data["vel_y"]] < 3.0:#velocity
                break
        if roll_flag == 0 and dev_flag == 0:
            break


        #try_acc = 1.0
    return try_acc,steer,tmp_state_vec,tmp_a_vec

def direct_emergency_cost(state,acc,steer,Direct,trainHP,roll_var,planningState = None):
    #assuming that if the next state is stable it is possible to stay stable by applying constant steering
    #next_vehicle_state,rel_pos,rel_ang = Direct.predict_one(state.Vehicle.values,[acc,steer])
    emergencyState = agent.State()
    emergencyState.Vehicle = actions.emergency_step(state.Vehicle,acc,steer,Direct)
    #if(Direct.check_stability1(emergencyState.Vehicle.values,factor = 0.5 if planningState.last_emergency_action_active else 1.0)):
    #    cost = 0
    #else:
    #    cost = 200
    state.env[1] = lib.find_index_on_path(state.env[0],emergencyState.Vehicle.abs_pos) 
    roll_flag,dev_flag = Direct.check_stability2(emergencyState.Vehicle,state.env,trainHP.max_plan_deviation,roll_var,0.5 if planningState.last_emergency_action_active else trainHP.stabilize_factor)
    if roll_flag == 0 and dev_flag== 0:#regular policy and emergency policy are ok:
        cost = 0
    else:
        cost = 200

    S_Vehicle = copy.deepcopy(state.Vehicle)
    tmp_state_vec,tmp_a_vec = [S_Vehicle]*5,[[acc,steer]]*5#tmp
    return cost, tmp_state_vec,tmp_a_vec

def emergency_cost(state,acc,steer,nets,trainHP,roll_var,Direct = None):
    #S.Vehicle = tmp_state_vec[0]
    S_Vehicle = copy.deepcopy(state.Vehicle)
    acc_command,steer_command = acc,steer
    tmp_a_vec = [[acc,steer]]#this action will operate on the saved state if not fail
    tmp_state_vec = []
                    
    for i in range(0,trainHP.rollout_n):#n times       
        if  roll_var < trainHP.const_var:
            roll_var += trainHP.one_step_var
        S_Vehicle = actions.step(S_Vehicle,acc_command,steer_command,nets.TransNet if not trainHP.direct_predict_active else Direct ,trainHP)                   
        tmp_state_vec.append(S_Vehicle) 
        steer_command = actions.emergency_steer_policy(S_Vehicle,state.env,trainHP,SteerNet =  nets.SteerNet if nets.steer_net_active else None)
        acc_command = actions.emergency_acc_policy()#always -1
        tmp_a_vec.append([acc_command,steer_command])
        
        state.env[1] = lib.find_index_on_path(state.env[0],S_Vehicle.abs_pos) 
        emergency_roll_flag,emergency_dev_flag = actions.check_stability(S_Vehicle,state.env,trainHP,roll_var = roll_var,max_plan_roll = trainHP.max_plan_roll,max_plan_deviation = trainHP.max_plan_deviation)
        if emergency_roll_flag != 0 or emergency_dev_flag != 0:
            break      
        if S_Vehicle.values[trainHP.vehicle_ind_data["vel_y"]] < 2.0:#velocity
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


def comp_MB_action(nets,state,acc,steer,trainHP,planningState = None, Direct = None):
    #print("trust_T:",planningState.trust_T)
    
    #print("planningState.last_emergency_action_active:",planningState.last_emergency_action_active)
    if planningState.last_emergency_action_active:
        planningState.trust_T = 0
    max_plan_roll = trainHP.max_plan_roll
    max_plan_deviation = trainHP.max_plan_deviation
    
    emergency_action_active = trainHP.emergency_action_flag#initialize to true just if in emergency mode
    #print("___________________new acc compution____________________________")
    state_env =  state.env 
    StateVehicle_vec = [state.Vehicle] 
    actions_vec = [[acc,steer]]
    StateVehicle_emergency_vec= [state.Vehicle]
    actions_emergency_vec = [[acc,steer]]
    
    
    if trainHP.direct_stabilize:
        emergencyState = copy.deepcopy(state)
        
    #emergency_action_active = False#tmp
    #delta_var = 0.002
    #init_var = 0.0#uncertainty of the roll measurment
    #one_step_var = 0.01#roll variance after a single step
    #const_var = 0.05#roll variance at the future states, constant because closed loop control?

    
    #stability at the initial state:
    
        if not trainHP.direct_constrain:
            current_roll_flag,current_dev_flag = actions.check_stability(state.Vehicle,state.env,trainHP,roll_var = planningState.var[0],max_plan_roll = max_plan_roll,max_plan_deviation = max_plan_deviation)
        else:
            current_roll_flag,current_dev_flag = Direct.check_stability2(state.Vehicle,state.env,trainHP.max_plan_deviation,planningState.var[0],0.5 if planningState.last_emergency_action_active else trainHP.stabilize_factor)
            if current_roll_flag:
                 print("caution - not stable---------------------")
            #roll_flag,dev_flag = 0,0#tmp
    if  trainHP.emergency_action_flag and planningState.trust_T == 0:#more accurate: trust_T == -1
        roll_flag,dev_flag = current_roll_flag,current_dev_flag
    else:
        roll_flag,dev_flag = 0,0
    if roll_flag == 0 and dev_flag == 0:     
        #predict the next unavoidable state (actions already done):  
        state.Vehicle = actions.step(state.Vehicle,acc,steer,nets.TransNet if not trainHP.direct_predict_active else Direct,trainHP)
  
        StateVehicle_vec.append(state.Vehicle)
        if trainHP.direct_stabilize and planningState.trust_T == 0:
            emergencyState.Vehicle = actions.emergency_step(state.Vehicle,acc,steer,Direct)
            StateVehicle_emergency_vec.append(emergencyState.Vehicle)
        else:
            StateVehicle_emergency_vec.append(state.Vehicle)

        state.env[1] = lib.find_index_on_path(state.env[0],state.Vehicle.abs_pos)   
        #roll_var += trainHP.one_step_var
        #stability at the next state (unavoidable state):
        if  trainHP.emergency_action_flag and planningState.trust_T == 0:
            if not trainHP.direct_constrain:
                roll_flag,dev_flag = actions.check_stability(state.Vehicle,state.env,trainHP,roll_var = planningState.var[1],max_plan_roll = max_plan_roll,max_plan_deviation = max_plan_deviation)
            else:
                roll_flag,dev_flag = Direct.check_stability2(state.Vehicle,state.env,trainHP.max_plan_deviation,planningState.var[1],0.5 if planningState.last_emergency_action_active else trainHP.stabilize_factor)
            #roll_flag,dev_flag = 0,0#tmp
        else:
            roll_flag,dev_flag = 0,0
        if roll_flag == 0 and dev_flag == 0:#first step was Ok
            next_acc,next_steer,tmp_state_vec,tmp_a_vec = driving_action(state,nets,trainHP,Direct,planningState =planningState)
            StateVehicle_vec += tmp_state_vec
            actions_vec += tmp_a_vec
            if trainHP.emergency_action_flag:
                #
                #check if the emergency policy is safe after applying the action on the first step
                if trainHP.direct_stabilize:
                    if planningState.trust_T < len(StateVehicle_vec):
                        #print("in emergency len(StateVehicle_vec):",len(StateVehicle_vec))
                        #check emergency cost after the trust range:
                        if planningState.trust_T > 0:
                            emergencyState.Vehicle = StateVehicle_vec[planningState.trust_T]
                            a,s = actions_vec[planningState.trust_T][0],actions_vec[planningState.trust_T][1]
                            roll_var = planningState.var[planningState.trust_T]
                        else:
                            a,s = next_acc,next_steer
                            roll_var = planningState.var[1]

                            
                        #else: emergencyState.Vehicle already computed

                        e_cost,tmp_state_vec,tmp_a_vec = direct_emergency_cost(emergencyState,a,s,Direct,trainHP,roll_var,planningState =planningState)
                    
                        if e_cost >= trainHP.max_cost and next_acc > -1.0: 
                            print("try to stabilize by braking")
                            next_acc = -1.0
                            e_cost,tmp_state_vec,tmp_a_vec = direct_emergency_cost(emergencyState,next_acc,next_steer,Direct,trainHP,roll_var,planningState =planningState)
                    else:
                        e_cost = 0

                else:
                    e_cost,tmp_state_vec,tmp_a_vec = emergency_cost(state,next_acc,next_steer,nets,trainHP,planningState.var[1],Direct)
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

        ##else:# first step will cause fail
            ##print("unavoidable fail - after first step")
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
        next_steer = actions.emergency_steer_policy(state.Vehicle if not trainHP.direct_stabilize else emergencyState.Vehicle,
                                                    state.env,trainHP,SteerNet =  nets.SteerNet if nets.steer_net_active else None) #steering from the next state, send it to the vehicle at the next state
        
        
        next_acc  = actions.emergency_acc_policy()#always -1
        ##print("emergency policy is executed!")

    if state.Vehicle.values[trainHP.vehicle_ind_data["vel_y"]]<trainHP.prior_safe_velocity:
        next_acc = 1.0

    #if state.Vehicle.values[trainHP.vehicle_ind_data["vel_y"]] > 5:
    #    next_acc = 0.0

    #next_steer = 0.4
    return next_acc,next_steer,StateVehicle_vec,actions_vec,StateVehicle_emergency_vec,actions_emergency_vec,emergency_action_active
    #return next_acc,next_steer,planningData,roll_flag,dev_flag