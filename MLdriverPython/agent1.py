

def compute_Qa(s,a,s_next,Agent): #roll out from s_next up to the target point

    return Qa

def update_Q(rand_state, rand_a, rand_reward, rand_next_state,rand_end,net,HP,comp_analytic_acceleration = None):
    rand_next_targetQa = net.get_targetQa(rand_next_state,rand_next_a)#like in DQN

    rand_targetQa = []
    for i in range(len(rand_state)):
        if rand_end[i] == False:
            rand_targetQa.append(rand_reward[i] + HP.gamma*rand_next_targetQa[i])#DQN  
        else:
            rand_targetQa.append([rand_reward[i]])

    rand_targetQa = []
    for i in range(len(rand_state)):
        rand_targetQa.append(compute_Qa(s,a,s_next,Agent))


    #update critic:
    net.Update_critic(rand_state,rand_a,rand_targetQa)#compute Qa(state,a) and minimize loss (Qa - targetQa)^2
    #Qa = net.get_Qa(rand_state,rand_a)
    #critic_loss = net.get_critic_loss(rand_state,rand_a,rand_targetQa)
    #print("critic_loss:",critic_loss)
    
    #update actor
    pred_action = net.get_actions(rand_state)#predicted action from state
    
    net.Update_actor(rand_state,pred_action)
    net.update_targets()
    return #critic_loss, Qa#temp


class MB_MF_Agent:
    def __init__(self,HP):
        MB_Agent = agent.Agent(HP)
