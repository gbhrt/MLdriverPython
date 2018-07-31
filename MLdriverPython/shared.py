import threading
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
        self.exit = False
        self.train = False
