import library as lib
import aggent_lib as a_lib
import classes
import planner
import enviroment_lib as env_lib
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import random
import json

def init_net_analytic(envData,net,save_file_path,restore_file_path,create_data_flag = True):
    buffer_size = 200000
    batch_size = 64
    num_train = 100000000

    stop = []
    command = []
    #lib.wait_for(stop,command)#wait for "enter" in another thread - then stop = true

    buffer = a_lib.Replay(buffer_size)
    if create_data_flag:
        #create data:
        pl = planner.Planner("dont_connect")
    
        while len(buffer.memory) < buffer.memory_size:# and stop == [False]
            pl.load_path(1000,source = "create_random")
            
            pl.external_update_vehicle([0,0,0], [0,0,0],0)
            pl.new_episode()#compute path in current vehicle position
            #print(pl.in_vehicle_reference_path.distance)
            #plt.figure(1)
            #plt.plot(np.array(pl.in_vehicle_reference_path.position)[:,0],np.array(pl.in_vehicle_reference_path.position)[:,1])
            #plt.figure(2)
            #plt.plot(pl.in_vehicle_reference_path.distance,pl.in_vehicle_reference_path.velocity_limit)
            #plt.plot(pl.in_vehicle_reference_path.distance,pl.in_vehicle_reference_path.analytic_velocity)
            #plt.plot(pl.in_vehicle_reference_path.distance,(np.array(pl.in_vehicle_reference_path.curvature))*100.0)
            #plt.show()
            
            for i in range(len(pl.in_vehicle_reference_path.position)-500):#up to the end of the path
                if stop == [True]:
                    break
                pl.external_update_vehicle(pl.in_vehicle_reference_path.position[i], pl.in_vehicle_reference_path.angle[i],pl.in_vehicle_reference_path.analytic_velocity[i])
                local_path = pl.get_local_path(send_path = False,num_of_points = int(envData.feature_points/0.05)+10)#
                #print(local_path.distance)
                #print(local_path.curvature)
                state = env_lib.get_state(pl = pl,local_path = local_path,num_points = envData.feature_points,distance_between = envData.distance_between_points,\
                    max_velocity = envData.max_velocity,max_curvature = envData.max_curvature)
                #print(state)
                v1 = pl.in_vehicle_reference_path.analytic_velocity[i]
                v2 = pl.in_vehicle_reference_path.analytic_velocity[i+1]
                d = pl.in_vehicle_reference_path.distance[i+1] - pl.in_vehicle_reference_path.distance[i]
                acc = (v2**2 - v1**2 )/(2*d)#acceleration [m/s^2]
                action =  [np.clip(acc/8,-1,1)]
                buffer.add((state,action))
                if len(buffer.memory) % 100 == 0:
                    print(len(buffer.memory))

        buffer.save(save_file_path,"analytic_data")
    else:
        buffer.restore(restore_file_path,"analytic_data")

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
    for i in range(0,len(buffer.memory) - 500,500):
        buff = buffer.memory[i:i+500]
        state_batch = []
        action_batch =[]
        for mem in buff:
            state_batch.append(mem[0])
            action_batch.append(mem[1])

        a = net.get_actions(state_batch)
        plt.plot(action_batch,'o')
        #plt.plot(a,'o')
        plt.show()
    #train:
    random.seed(1234)
    random.shuffle(buffer.memory)
    buffer.memory = buffer.memory[:buffer_size]
    print("buffer len: ",len(buffer.memory))
    test_buffer = a_lib.Replay(buffer_size)
    train_buffer = a_lib.Replay(buffer_size)
    test_buffer.memory = buffer.memory[:int(0.5*len(buffer.memory))]
    train_buffer.memory = buffer.memory[int(0.5*len(buffer.memory)):]

    
    #state_batch,action_batch = buffer.sample(1000)
    #for i in range(1000):
    #    if action_batch[i][0] > 0.85 and action_batch[i][0] < 0.95:
    #        plt.plot(state_batch[i],'o')
    #        plt.plot(action_batch[i],'o')
    #        plt.show()

    loss_vec = train(net,train_buffer,num_train,batch_size)
    net.save_model(save_file_path)
    #"1e4.txt"
    #"1e4_no_norm.txt"
    #with open(save_file_path+"1e4_no_reg.txt", 'w') as f:
    #    json.dump(loss_vec,f)

    print("test loss:",test(net,test_buffer))
    test_action_diff(net,train_buffer)

def test_action_diff(net,buffer):
    state_batch,action_batch = buffer.sample(len(buffer.memory))
    a = net.get_actions(state_batch)
    plt.figure(1)
    plt.plot(action_batch,'o')
    plt.plot(a,'o')
    plt.figure(2)
    #diff_a = [action_batch[i][0] -a[i][0] for i in range(min(len(action_batch),len(a)))]
    #plt.plot(diff_a,'o')
    #plt.show()

def test(net,buffer):
    state_batch,action_batch = buffer.sample(len(buffer.memory))
    return net.get_analytic_actor_loss(state_batch,action_batch)

def train(net,buffer,num_train,batch_size):
    stop = []
    command = []
    lib.wait_for(stop,command)#wait for "enter" in another thread - then stop = true
    loss_vec = []
    for i in range(num_train):
        if stop == [True]:
            break
        state_batch,action_batch = buffer.sample(batch_size)
        #plt.plot(state_batch[0])
        #plt.plot(state_batch[1])
        #plt.plot(state_batch[2])
        #plt.show()
        net.Update_analytic_actor(state_batch,action_batch)

        if i % 100 == 0:
            loss = float(net.get_analytic_actor_loss(state_batch,action_batch))
            a = net.get_actions(state_batch)
            loss_vec.append(loss)
            print("loss:",loss)#,"action analytic:",action_batch,"action:",a)
        if command == [b'1']:
            plt.plot(loss_vec)
            plt.plot(lib.running_average(loss_vec,50))
            plt.show()
            test_action_diff(net,buffer)
            command[0] = b'0'
    return loss_vec
    
