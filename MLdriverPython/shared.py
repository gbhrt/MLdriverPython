import threading
import classes


class guiShared():#classes.planningData
    def __init__(self):
        #super().__init__()
        self.Lock = threading.Lock()
        self.request_exit = False
        self.exit = False
        self.pause_after_episode_flag = False
        self.evaluate = False
        
        self.state = None
        self.steer = None
        self.action = []

        self.steering_target = None
        self.max_roll = 0
        self.max_time = 30

        self.roll = []
        self.real_path= []

        self.episodes_data = []#reward for each episode
        self.episodes_fails = []#final states - 0 ok, 1 failed, 2 emergency activated
        self.update_episodes_flag = False
        self.update_data_flag = False
        self.restart()
        return

    def restart(self):
        self.planningData = classes.planningData()


class trainShared:
    def __init__(self):
        self.Lock = threading.Lock()
        self.algorithmIsIn = threading.Event()
        self.request_exit = False
        self.exit = False
        self.train = False

class simulatorShared:
    def __init__(self):
        self.Lock = threading.Lock()
        self.algorithmIsIn = threading.Event()
        self.exit = False
        self.connected = None
        self.end_connection_flag = False
        self.vehicle = classes.Vehicle()
        self.commands = [0,0]#commands[0] = steering,commands[1] = acceleration 
        self.send_commands_flag = False
        self.path = classes.Path()
        self.send_path_flag = False
        self.reset_position_flag = False
        
