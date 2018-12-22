import threading
import classes
class guiShared:
    def __init__(self):
        self.Lock = threading.Lock()
        self.exit_program = False
        self.path = None
        self.state = None
        self.steer = None
        self.predicded_path = None
        self.steering_target = None
        #self.start_draw
        return

class trainShared:
    def __init__(self):
        self.Lock = threading.Lock()
        self.algorithmIsIn = threading.Event()
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
        
