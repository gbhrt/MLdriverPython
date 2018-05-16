import os
class HyperParameters:
    def __init__(self):
        #self.epsilon_start = 1.0
        #self.epsilon = 0.1
        self.gamma = 0.99
        self.tau = 0.001 #how to update target network compared to Q network
        self.num_of_runs = 100000
        self.alpha_actor = 0.0001# for Pi 1e-5 #learning rate
        self.alpha_critic = 0.001#for Q
        self.alpha_analytic_actor = 1e-4
        self.alpha_analytic_critic = 1e-5
        self.batch_size = 64
        self.replay_memory_size = 100000
        self.train_num = 5# how many times to train in every step

        #########################
        self.gym_flag = False
        self.render_flag = True
        self.plot_flag = True
        self.restore_flag = False
        self.skip_run = False
        self.reset_every = 3
        self.save_every = 100
        self.seed = 1234
        self.save_name = "model11" 
        self.save_file_path = os.getcwd()+ "\\files\\models\\DDPG\\"+self.save_name+"\\"
        #7 INIT
        self.restore_name = "model11" # m
        self.restore_file_path = os.getcwd()+ "\\files\\models\\DDPG\\"+self.restore_name+"\\"

        self.run_data_file_name = 'running_record1'