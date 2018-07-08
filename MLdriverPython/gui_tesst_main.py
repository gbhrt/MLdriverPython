import gui
import threading
import time
import main_program
import main_model_based
import gui_shared

class programThread (threading.Thread):
   def __init__(self,shared):
        threading.Thread.__init__(self)
        self.shared = shared
      
   def run(self):
        print ("Starting " + self.name)
        main_model_based.run(shared)
        #main_program.run(shared)
        print ("Exiting " + self.name)

#def print_time(threadName, counter, delay):
#   while counter:
#      if exitFlag:
#         threadName.exit()
#      time.sleep(delay)
#      print ("%s: %s" % (threadName, time.ctime(time.time())))
#      counter -= 1

if __name__ == "__main__": 

    shared = gui_shared.shared()
    # Create new threads
    programThread = programThread(shared)
    # Start new Threads
    programThread.start()

    #start the gui:
    gui.start_gui(shared)

    print ("Exiting Main Thread")

