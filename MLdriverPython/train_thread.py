import threading
from model_based_net import model_based_network
import agent_lib as pLib
import time
import random
import numpy as np

def train(nets,Replay,trainHP,HP,trainShared):
    while not trainShared.request_exit:
        if len(Replay.memory) > 2 and HP.train_flag:############ len(Replay.memory) > HP.batch_size and 
            trainShared.train = True
            last_save_time = time.time()
            train_count = 0
            break
        time.sleep(0.01)
    while trainShared.train:

        trainShared.algorithmIsIn.wait()#wait that the other thread enabled the lock
        with trainShared.Lock:
            #rand_state, rand_a, rand_next_state, rand_end,_ = Replay.sample(HP.batch_size)
            #batch_X, batch_Y_, _,_ = Replay.sample(trainHP.batch_size)
            #sample_indexes = random.sample(range(1, len(Replay.memory) - 1),trainHP.batch_size)
            TransNet_X,TransNet_Y_ = [],[]
            AccNet_X,AccNet_Y_ = [],[]
            SteerNet_X,SteerNet_Y_ = [],[]
            cnt = 0
            #cliped_batch_size = np.clip(trainHP.batch_size,0,len(Replay.memory))
            max_cnt = 0
            while cnt < trainHP.batch_size and max_cnt < 100:#cliped_batch_size:
                ind = random.randint(0,len(Replay.memory) - 2)
                if Replay.memory[ind][3] or Replay.memory[ind+1][4] : # done flag, time_error
                    max_cnt+=1#if every
                    continue
                #print("ind:",ind)
                vehicle_state = Replay.memory[ind][0]
                action = Replay.memory[ind][2]
                vehicle_state_next = Replay.memory[ind+1][0]
                rel_pos = Replay.memory[ind+1][1]
                if nets.trans_net_active:
                    TransNet_X.append(vehicle_state+action)
                    TransNet_Y_.append([vehicle_state_next[i] - vehicle_state[i] for i in range(len(vehicle_state_next))] +rel_pos)

                if nets.acc_net_active:
                    AccNet_X.append(vehicle_state+[action[1]]+[vehicle_state_next[trainHP.vehicle_ind_data['roll']]])
                    AccNet_Y_.append(action[0])
                if nets.steer_net_active:
                    SteerNet_X.append(vehicle_state+[action[0]]+[vehicle_state_next[trainHP.vehicle_ind_data['roll']]])
                    SteerNet_Y_.append(action[1])

                cnt+=1

            #update neural networs:
            #pLib.model_based_update(rand_state, rand_a, rand_next_state,rand_end,net,HP,env
            with nets.transgraph.as_default():                #
                #try:
                if nets.trans_net_active:
                    nets.TransNet.train_on_batch(np.array(TransNet_X),np.array(TransNet_Y_))
                #except:
                #    print("TransNet_Y_:",TransNet_Y_)
                if nets.steer_net_active:
                    nets.SteerNet.train_on_batch(np.array(SteerNet_X),np.array(SteerNet_Y_))
                if nets.acc_net_active:
                    nets.AccNet.train_on_batch(np.array(AccNet_X),np.array(AccNet_Y_))
                if train_count % 1000 == 0:
                    print("train:",train_count)
                    if nets.trans_net_active:
                        print("Trans loss:",nets.TransNet.evaluate(np.array(TransNet_X),np.array(TransNet_Y_)))
                    if nets.steer_net_active:
                        print("Steer loss:",nets.SteerNet.evaluate(np.array(SteerNet_X),np.array(SteerNet_Y_)))
                    #print(nets.SteerNet.evaluate(np.array(SteerNet_X),np.array(SteerNet_Y_)))
            #    #X,Y_ = env.create_XY_(rand_state, rand_a, rand_next_state)
            #    print("loss:",float(net.get_loss(batch_X,batch_Y_)))
                #print("loss:",float(net.get_loss(X,Y_)))
                if time.time() - last_save_time > HP.save_every_time*60:
                    nets.save_all(HP.save_file_path,HP.net_name)
                    Replay.save(HP.save_file_path)
                    last_save_time = time.time()


                #print("update model")
                
            
            train_count+=1

            

            
    
    trainShared.exit = True
        

            #net.save_model(HP.save_file_path)
            #Replay.save(HP.save_file_path)





class trainThread (threading.Thread):
    def __init__(self,nets,Replay,trainHP,HP,trainShared):
        threading.Thread.__init__(self)
        self.nets = nets
        self.Replay = Replay
        self.trainHP = trainHP
        self.HP = HP
        self.trainShared = trainShared
      
    def run(self):
        print ("Starting " + self.name)
        train(self.nets,self.Replay,self.trainHP,self.HP,self.trainShared)
        print ("Exiting " + self.name)



    
        





def MF_train(nets,Replay,trainHP,HP,trainShared,MF_net):
    while not trainShared.request_exit:
        if len(Replay.memory) > 2 and HP.train_flag:############ len(Replay.memory) > HP.batch_size and 
            trainShared.train = True
            last_save_time = time.time()
            train_count = 0
            break
        time.sleep(0.01)

    while trainShared.train:
        trainShared.algorithmIsIn.wait()#wait that the other thread enabled the lock
        with trainShared.Lock:
            #rand_state, rand_a, rand_next_state, rand_end,_ = Replay.sample(HP.batch_size)
            #batch_X, batch_Y_, _,_ = Replay.sample(trainHP.batch_size)
            #sample_indexes = random.sample(range(1, len(Replay.memory) - 1),trainHP.batch_size)
            TransNet_X,TransNet_Y_ = [],[]
            AccNet_X,AccNet_Y_ = [],[]
            SteerNet_X,SteerNet_Y_ = [],[]
            cnt = 0
            #cliped_batch_size = np.clip(trainHP.batch_size,0,len(Replay.memory))
           # print("Replay.memory:",Replay.memory)
            max_cnt = 0
            while cnt < trainHP.batch_size and max_cnt < 100:#cliped_batch_size:
                ind = random.randint(0,len(Replay.memory) - 2)
                if Replay.memory[ind][3] or Replay.memory[ind+1][4] : # done flag, time_error
                    max_cnt+=1#if every
                    continue
                #print("ind:",ind)
                vehicle_state = Replay.memory[ind][0]
                action = Replay.memory[ind][2]
                vehicle_state_next = Replay.memory[ind+1][0]
                rel_pos = Replay.memory[ind+1][1]

                TransNet_X.append(vehicle_state+action)
                TransNet_Y_.append([vehicle_state_next[i] - vehicle_state[i] for i in range(len(vehicle_state_next))] +rel_pos)

                
                AccNet_X.append(vehicle_state+[action[1]]+[vehicle_state_next[trainHP.vehicle_ind_data['roll']]])
                AccNet_Y_.append(action[0])

                SteerNet_X.append(vehicle_state+[action[0]]+[vehicle_state_next[trainHP.vehicle_ind_data['roll']]])
                SteerNet_Y_.append(action[1])

                cnt+=1

            #update neural networs:
            #pLib.model_based_update(rand_state, rand_a, rand_next_state,rand_end,net,HP,env
            with nets.transgraph.as_default():
                #
                #try:
                nets.TransNet.train_on_batch(np.array(TransNet_X),np.array(TransNet_Y_))
                #except:
                #    print("TransNet_Y_:",TransNet_Y_)
                nets.SteerNet.train_on_batch(np.array(SteerNet_X),np.array(SteerNet_Y_))
                #nets.AccNet.train_on_batch(np.array(AccNet_X),np.array(AccNet_Y_))
                if train_count % 1000 == 0:
                    print("train:",train_count)
                    print("Trans loss:",nets.TransNet.evaluate(np.array(TransNet_X),np.array(TransNet_Y_)))
                    print("Steer loss:",nets.SteerNet.evaluate(np.array(SteerNet_X),np.array(SteerNet_Y_)))
                    #print(nets.SteerNet.evaluate(np.array(SteerNet_X),np.array(SteerNet_Y_)))
            #    #X,Y_ = env.create_XY_(rand_state, rand_a, rand_next_state)
            #    print("loss:",float(net.get_loss(batch_X,batch_Y_)))
                #print("loss:",float(net.get_loss(X,Y_)))
                if time.time() - last_save_time > HP.save_every_time*60:
                    nets.save_all(HP.save_file_path)
                    Replay.save(HP.save_file_path)
                    last_save_time = time.time()
            train_count+=1
    trainShared.exit = True

class MFtrainThread (threading.Thread):
   def __init__(self,nets,Replay,trainHP,HP,trainShared,MF_net):
        threading.Thread.__init__(self)
        self.nets = nets
        self.Replay = Replay
        self.trainHP = trainHP
        self.HP = HP
        self.trainShared = trainShared
        self.MF_net = MF_net
      
   def run(self):
        print ("Starting " + self.name)
        train(self.nets,self.Replay,self.trainHP,self.HP,self.trainShared,self.MF_net)
        print ("Exiting " + self.name)