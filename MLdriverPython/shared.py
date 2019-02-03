import threading
import classes


class guiShared():#classes.planningData
    def __init__(self):
        #super().__init__()
        self.Lock = threading.Lock()
        self.request_exit = False
        self.exit = False
        self.pause_after_episode_flag = False
        
        self.state = None
        self.steer = None

        self.steering_target = None
        self.max_roll = 0
        self.max_time = 100

        self.roll = []
        self.real_path= []
        
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
        
