from planner import Planner
import numpy as np
import random
import time
from library import wait_for
from linear_reg_Qlearn import Network

stop = []


plan = Planner()
plan.simulator.set_address("10.2.1.111",5007)# if run in linux # "10.0.17.74"
plan.start_simple()
plan.init_spaces()

net = Network(1,plan.action_space.data)
print("Planner ready")
print("Network ready")
net.restore()

epsilon = 0.1
gamma = 0.95
alpha = 0.1
num_of_runs = 10
step_time = 0.5
#Q = np.zeros ([len(plan.state_space.data),len(plan.action_space.data)])
print("start")
wait_for(stop)#wait for "enter" in annother thread - then stop = true
for i in range(num_of_runs): #number of runs
    if stop:
        break
    s =  plan.dist_from_target() #plan.get_state_space_index()
    done = False
    while not done and not stop: #until game over, or user stop
         all_Qs = net.get_Q(s)
         if random.random() < epsilon:
            a = random.randint(0,len(plan.action_space.data)-1)
            print("random a: ",a)
         else:
             #a = np.argmax(Q[s,:])
             print("all_Qs: ",all_Qs)
             a = np.argmax(all_Qs)
             print("best a: ",a)
         #step:
         plan.simulator.send_drive_commands(plan.simulator.vehicle.velocity + plan.action_space.data[a],0)
         time.sleep(step_time)
         plan.simulator.get_vehicle_data()
         s_next = plan.dist_from_target() # plan.get_state_space_index()
         r = plan.get_reward(s,s_next)
         print("r: :",r)
         if plan.dist_from_target() < 5:
             done = True
             plan.restart()
         #end step
         Q_corrected = np.copy(all_Qs)
         next_Q = net.get_Q(s_next)
         Q_corrected[0][a] = r +gamma*np.max(next_Q)
         print("Q_corrected: ",Q_corrected)
         W1,b1 = net.get_par()
         print("W1_1: ",W1,"b1_1: ",b1)
         net.update_sess(s,Q_corrected)
        
         W1,b1 = net.get_par()
         print("W1_1: ",W1,"b1: ",b1)
         print("s: ",s, " s_next: ",s_next)
         s = s_next
         net.save_model()

plan.end()
  #Q[s,a] = Q[s,a] + alpha*(r + gamma * np.max(Q[s_next,:]) - Q[s,a])
         #print(Q,"\n")
      #if random.random() < epsilon:
      #    a = random.choice(plan.action_space)
      #else:
      #    a = np.argmax(Q[s,:])
      #s_n, r, done, _ = env.step(a)
      #Q[s,a] = Q[s,a] + alpha*(r + gamma * np.max(Q[s_n,:]) - Q[s,a])
      #s = s_n