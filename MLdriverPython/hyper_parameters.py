import os
class HyperParameters:
    def __init__(self):
        #self.epsilon_start = 1.0
        #self.epsilon = 0.1
        self.gamma = 0.99
        self.tau = 0.001 #how to update target network compared to Q network
        self.num_of_runs = 100000
        self.alpha_actor = 0.0001# for Pi 1e-5 #learning rate
        self.alpha_critic = 0.001#for Q 0.001
        self.alpha_analytic_actor = 1e-5#1e3 - too high, loss stop at 0.1
        self.alpha_analytic_critic = 1e-4
        self.batch_size = 64
        self.replay_memory_size = 1000000
        self.train_num = 2# how many times to train in every step
        self.sample_ratio = 1.0
        self.epsilon = 0.1
        #########################
        self.evaluation_flag = False
        self.reduce_vel = 0.0
        self.add_feature_to_action  = False
        self.analytic_action = False
        self.train_flag =True
        self.noise_flag = True
        self.always_no_noise_flag = False
        self.evaluation_every = 1000000
        self.test_same_path = False
        self.run_same_path = False
        self.conv_flag = False
        self.gym_flag = False
        self.render_flag = True
        self.plot_flag = True
        self.restore_flag = False
        self.skip_run = False
        self.reset_every = 3
        self.save_every = 50
        self.save_every_train_number = 2500
        self.seed = [1111]#,1112,1113,1114,1115]
        self.save_name ="REVO"
        self.save_file_path = os.getcwd()+ "\\files\\models\\paper_fix\\"+self.save_name+"\\"

        self.restore_name = "REVO"
        self.restore_file_path = os.getcwd()+ "\\files\\models\\paper_fix\\"+self.restore_name+"\\"

        if self.always_no_noise_flag:
            self.noise_flag = False

class ModelBasedHyperParameters:#global settings of the program.
    def __init__(self):
       
        self.program_mode =  "train_in_env"#"test_net_performance" train_in_env, test_net_performance, test_actions
        if self.program_mode == "test_net_performance" or self.program_mode == "test_actions":
            self.gui_flag = False
        else:
            self.gui_flag = True
        #########################
        self.train_flag = True
        #self.noise_flag = True
        #self.always_no_noise_flag = False
        self.analytic_action = False
       # self.zero_noise_every = 1
        self.test_same_path = False
        self.run_same_path = False
       # self.conv_flag = False
        self.gym_flag = False
        self.render_flag = True
        self.plot_flag = True
        self.restore_flag = True
        self.skip_run = False
        self.reset_every = 3
        self.save_every = 100
        self.save_every_time = 5000 #minutes
        self.seed = 1111
        self.save_name = "collect_data_2"
        self.save_file_path = os.getcwd()+ "\\files\\models\\model_based\\"+self.save_name+"\\"
        self.restore_name = "collect_data_2"#
        self.restore_file_path = os.getcwd()+ "\\files\\models\\model_based\\"+self.restore_name+"\\"

       