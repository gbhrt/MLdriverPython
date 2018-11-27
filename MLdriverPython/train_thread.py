import threading
from model_based_net import model_based_network
import agent_lib as pLib
import time

def train(net,Replay,HP,env,trainShared):
    while not trainShared.exit:
        if len(Replay.memory) > 2 and HP.train_flag:############ len(Replay.memory) > HP.batch_size and 
            trainShared.train = True
            last_save_time = time.time()
            train_count = 0
            break
        time.sleep(0.01)

    while trainShared.train:
       
        # t = time.clock()

        trainShared.algorithmIsIn.wait()#wait that the other thread enabled the lock
        with trainShared.Lock:
            rand_state, rand_a, rand_next_state, rand_end,_ = Replay.sample(HP.batch_size)
            
            #update neural networs:
            pLib.model_based_update(rand_state, rand_a, rand_next_state,rand_end,net,HP,env)
            train_count+=1

            if train_count % 100 == 0:
                print("train:",train_count)
            if time.time() - last_save_time > HP.save_every_time*60:
                net.save_model(HP.save_file_path)
                Replay.save(HP.save_file_path)
                last_save_time = time.time()

            #print("train:",train_count)

 
        

            #net.save_model(HP.save_file_path)
            #Replay.save(HP.save_file_path)





class trainThread (threading.Thread):
   def __init__(self,net,Replay,HP,env,trainShared):
        threading.Thread.__init__(self)
        self.net = net
        self.Replay = Replay
        self.HP = HP
        self.env = env
        self.trainShared = trainShared

      
   def run(self):
        print ("Starting " + self.name)
        train(self.net,self.Replay,self.HP,self.env,self.trainShared)
        print ("Exiting " + self.name)

