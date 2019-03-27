


def comp_MB_action(state, policyHP, model, acc, steer):
    planning_data = []

    state.abs_pos = [0,0,0]#initialize

    state = get_next_state(model.net,state,acc,steer)#predict the next unavoidable state (actions already done):

    steer_vec = comp_steer(state,policyHP,model)

    state_vec, trajectory, acc,steer,emergency = comp_acc(state,policyHP,model.net)


    steer = steer_vec[0]
    return acc,steer

