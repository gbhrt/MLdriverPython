import library as lib
import agent_lib as a_lib
import classes
import planner
import enviroment_lib as env_lib
import numpy as np
import matplotlib.pyplot as plt
import random
import json


def create_data(envData,net,save_file_path,restore_file_path,gamma,buffer_size,waitFor):
    buffer = a_lib.Replay(buffer_size)
    
    #create data:
    pl = planner.Planner("dont_connect")
    
    while len(buffer.memory) < buffer.memory_size:# and stop == [False]
        pl.load_path(9000,source = "create_random")
            
        pl.external_update_vehicle([0,0,0], [0,0,0],0)
        pl.new_episode()#compute path in current vehicle position

        for i in range(0,len(pl.in_vehicle_reference_path.position)-500):#,int(envData.distance_between_points/0.05) #up to the end of the path
            if waitFor.stop == [True]:
                break
            pl.external_update_vehicle(pl.in_vehicle_reference_path.position[i], pl.in_vehicle_reference_path.angle[i],pl.in_vehicle_reference_path.analytic_velocity[i])
            local_path = pl.get_local_path(send_path = False,num_of_points = int(envData.feature_points/0.05)+10)#

            state = env_lib.get_state(pl = pl,local_path = local_path,num_points = envData.feature_points,distance_between = envData.distance_between_points,\
                max_velocity = envData.max_velocity,max_curvature = envData.max_curvature)

            position_state = env_lib.choose_position_points(local_path,envData.feature_points,envData.distance_between_points)
            state_path = classes.Path()
            for j in range(0,len(position_state)-1,2):
                state_path.position.append([position_state[j],position_state[j+1],0.0])
            lib.comp_velocity_limit_and_velocity(state_path,init_vel = pl.simulator.vehicle.velocity, final_vel = 0)
            acc = state_path.analytic_acceleration[0]
            #print(acc)
            #if i % 1 == 0:
            #    plt.figure(1)
            #    plt.plot(np.array(state_path.position)[:,0],np.array(state_path.position)[:,1],'o')
            #    plt.figure(2)
            #    plt.plot(state_path.distance,state_path.analytic_velocity_limit,'o')
            #    plt.plot(state_path.distance,state_path.analytic_velocity,'o')
            #    plt.plot(state_path.distance,state_path.analytic_acceleration,'o')
            #    plt.show()
            action =  [np.clip(acc/8,-1,1)]
            #action2 = lib.comp_analytic_acceleration(state,envData.distance_between_points*envData.feature_points,envData.max_velocity)
            #######################
            #compute Q:
            rewards = []
            start_time = pl.in_vehicle_reference_path.analytic_time[i]
            Q = 0.0#sum of (reduced) rewards from current state until end of the path (in step time intervals)
            step_count = 0 
            for j in range(i,  len(pl.in_vehicle_reference_path.analytic_velocity)):
                if pl.in_vehicle_reference_path.analytic_time[j] - start_time < envData.step_time:#search the next time step
                    continue
                r = env_lib.get_reward(pl.in_vehicle_reference_path.analytic_velocity[j],envData.max_velocity,'ok')#assume vehicle always ok 
                rewards.append(r)
                Q += (gamma**step_count) * r
                step_count+=1
                if step_count == envData.max_episode_steps:
                    break
                start_time = pl.in_vehicle_reference_path.analytic_time[j]

            #print(Q)
            #print(rewards)
            ############################



            buffer.add((state,action,[Q]))
            if len(buffer.memory) % 100 == 0:
                print(len(buffer.memory))
        #print("new path")
    return buffer


def init_net_analytic(envData,net,save_file_path,restore_file_path,gamma,create_data_flag = True):
    buffer_size = 100000
    batch_size = 64
    num_train = 100000000

    waitFor = lib.waitFor()

    #if create_data_flag:
    #    buffer = create_data(envData,net,save_file_path,restore_file_path,gamma,buffer_size,waitFor)
    #    buffer.save(save_file_path,"analytic_data")
    #else:
    #    buffer = a_lib.Replay(buffer_size)
    #    buffer.restore(restore_file_path,"analytic_data")

    replay_buffer = a_lib.Replay(buffer_size)
    replay_buffer.restore(restore_file_path)

    buffer = a_lib.Replay(buffer_size)

    state_vec, a_vec,reward_vec, next_state_vec, end_vec = replay_buffer.sample(len(replay_buffer.memory))
    for state, a,reward in zip( state_vec, a_vec,reward_vec):
        buffer.add((state,a,reward))

    #for mem in buffer.memory:
    #    mem[0] = [-i for i in mem[0]]
    #with open(save_file_path+"1e4_no_reg.txt", 'r') as f:
    #    test_buffer1 = json.load(f)#
    #with open(save_file_path+"1e4.txt", 'r') as f:
    #    test_buffer2 = json.load(f)#

    #plt.plot(test_buffer1)
    #plt.plot(test_buffer2)
    #plt.show()
    #buffer.memory = buffer.memory[:500]
    #print(buffer.memory)
    #for i in range(0,len(buffer.memory) - 500,500):
    #    buff = buffer.memory[i:i+500]
    #    state_batch = []
    #    action_batch =[]
    #    for mem in buff:
    #        state_batch.append(mem[0])
    #        action_batch.append(mem[1][0])
    #        print(action_batch)
    #    a = net.get_actions(state_batch)
    #    plt.plot(action_batch,'o')
    #    plt.plot(a,'o')
    #    plt.show()
    #train:
   # random.seed(1234)
    random.shuffle(buffer.memory)
    buffer.memory = buffer.memory[:buffer_size]
    print("buffer len: ",len(buffer.memory))
    test_buffer = a_lib.Replay(buffer_size)
    train_buffer = a_lib.Replay(buffer_size)
    train_buffer.memory = buffer.memory[:int(0.5*len(buffer.memory))]
    test_buffer.memory = buffer.memory[int(0.5*len(buffer.memory)):]
    

    

    actor_loss_vec = []
    #critic_loss_vec = []
    #for i in range(num_train):
    #   train_actor(net,train_buffer,10000,batch_size,actor_loss_vec,waitFor)
       #train_critic(net,train_buffer,10000,batch_size,critic_loss_vec,waitFor)
   # train_actor(net,train_buffer,10000000,batch_size,actor_loss_vec,waitFor)
    train_actor(net,train_buffer,1000000,batch_size,actor_loss_vec,waitFor)
    #"1e4.txt"
    #"1e4_no_norm.txt"
    #with open(save_file_path+"1e4_no_reg.txt", 'w') as f:
    #    json.dump(loss_vec,f)
    print("actor test loss:",test_actor_loss(net,test_buffer))
    #print("test critic loss:",test_critic_loss(net,test_buffer))
    test_action_diff(net,test_buffer)
    #test_Q_diff(net,test_buffer)

    net.copy_targets()#copy critic and actor networks to the target networks
    net.save_model(save_file_path)
   # test_action_diff(net,train_buffer)

def test_action_diff(net,buffer):
    state_batch,action_batch,Q_batch = buffer.sample(len(buffer.memory))
    a = net.get_actions(state_batch)
    plt.figure(1)
    plt.plot(action_batch,'o')
    plt.plot(a,'o')
    #plt.figure(2)
    #diff_a = [action_batch[i][0] -a[i][0] for i in range(min(len(action_batch),len(a)))]
    #plt.plot(diff_a,'o')
    plt.show()
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
def test_actor_loss(net,buffer):
    state_batch,action_batch,Q_batch = buffer.sample(len(buffer.memory))
    return net.get_analytic_actor_loss(state_batch,action_batch)
def test_critic_loss(net,buffer):
    state_batch,action_batch,Q_batch = buffer.sample(len(buffer.memory))
    return net.get_critic_loss(state_batch,action_batch,Q_batch)

def train_actor(net,buffer,num_train,batch_size,actor_loss_vec,waitFor):

    plt.ion()
    for i in range(num_train):
        if waitFor.stop == [True]:
            break
        state_batch,action_batch,Q = buffer.sample(batch_size)
        #plt.plot(state_batch[0])
        #plt.plot(state_batch[1])
        #plt.plot(state_batch[2])
        #plt.show()
        net.Update_analytic_actor(state_batch,action_batch)

        if i % 100 == 0:
            loss = float(net.get_analytic_actor_loss(state_batch,action_batch))
            #a = net.get_actions(state_batch)
            actor_loss_vec.append(loss)
            print("actor loss:",loss)#,"action analytic:",action_batch,"action:",a)
            plt.cla()
            plt.plot(actor_loss_vec)
            plt.plot(lib.running_average(actor_loss_vec,50))
            plt.draw()
            plt.pause(0.0001)

        if waitFor.command == [b'1']:
            plt.plot(actor_loss_vec)
            plt.plot(lib.running_average(actor_loss_vec,50))
            plt.show()
            test_action_diff(net,buffer)
            waitFor.command[0] = b'0'
    plt.ioff()        
    plt.show()
    return 

def train_critic(net,buffer,num_train,batch_size,critic_loss_vec,waitFor):
    for i in range(num_train):
        if waitFor.stop == [True]:
            break
        state_batch,action_batch,Q_batch = buffer.sample(batch_size)
        #plt.plot(state_batch[0])
        #plt.plot(state_batch[1])
        #plt.plot(state_batch[2])
        #plt.show()
        #net.Update_analytic_actor(state_batch,action_batch)
        net.Update_critic(state_batch,action_batch,Q_batch)

        if i % 100 == 0:
            loss = float(net.get_critic_loss(state_batch,action_batch,Q_batch))
            critic_loss_vec.append(loss)
            print("critic loss:",loss)#,"action analytic:",action_batch,"action:",a)
        if waitFor.command == [b'1']:
            plt.plot(critic_loss_vec)
            plt.plot(lib.running_average(critic_loss_vec,50))
            plt.show()
            test_Q_diff(net,buffer)
            waitFor.command[0] = b'0'
    return 
    
