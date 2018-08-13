from hyper_parameters import HyperParameters
from DDPG_net import DDPG_network
import enviroment1
import agent_lib as pLib
import time
import library as lib
import numpy as np
import matplotlib.pyplot as plt
import random

def test_Q_diff(net,buffer):
    state_batch,action_batch,Q_batch = buffer.sample(len(buffer.memory))
    #a = net.get_actions(state_batch)
    Q_predicted = net.get_Qa(state_batch,action_batch)
    plt.figure(1)
    plt.plot(Q_batch,'o')
    plt.plot(Q_predicted,'o')
    #plt.figure(2)
    #diff_a = [action_batch[i][0] -a[i][0] for i in range(min(len(action_batch),len(a)))]
    #plt.plot(diff_a,'o')
    plt.show()

if __name__ == "__main__": 
    waitFor = lib.waitFor()
    HP = HyperParameters()
    envData = enviroment1.OptimalVelocityPlannerData()
    net = DDPG_network(envData.observation_space.shape[0],envData.action_space.shape[0],envData.action_space.high[0],\
            HP.alpha_actor,HP.alpha_critic,HP.alpha_analytic_actor,HP.alpha_analytic_critic,tau = HP.tau,seed = HP.seed[0],feature_data_n = envData.feature_data_num, conv_flag = HP.conv_flag)  
    if HP.restore_flag:
        net.restore(HP.restore_file_path)

    Replay = pLib.Replay(HP.replay_memory_size)

    Replay.restore(HP.restore_file_path)
    #random.shuffle(Replay.memory)
    #Replay.memory = Replay.memory[:1000]
    critic_loss_vec = []
    start_time = time.time()
    for i in range(10000000):
        if waitFor.stop == [True]:
            break
        rand_state, rand_a, rand_reward, rand_next_state, rand_end = Replay.sample(HP.batch_size)
        critic_loss,Qa = pLib.DDPG(rand_state, rand_a, rand_reward, rand_next_state,rand_end,net,HP)
        if i % 10 == 0:
            critic_loss_vec.append(np.mean(Qa))
            print("critic_loss: ",critic_loss)
            print("time:",time.time() - start_time)
        if waitFor.command == [b'1']:
            plt.plot(critic_loss_vec)
            plt.plot(lib.running_average(critic_loss_vec,50))
            plt.show()
            #test_Q_diff(net,buffer)
            waitFor.command[0] = b'0'

    net.save_model(HP.save_file_path)