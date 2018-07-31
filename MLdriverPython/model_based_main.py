import gui
import threading
import time
import main_program
import model_based_run
import shared

class programThread (threading.Thread):
   def __init__(self,guiShared):
        threading.Thread.__init__(self)
        self.guiShared = guiShared
      
   def run(self):
        print ("Starting " + self.name)
        model_based_run.run(guiShared)
        print ("Exiting " + self.name)

#def print_time(threadName, counter, delay):
#   while counter:
#      if exitFlag:
#         threadName.exit()
#      time.sleep(delay)
#      print ("%s: %s" % (threadName, time.ctime(time.time())))
#      counter -= 1

if __name__ == "__main__": 

    guiShared = shared.guiShared()

    # Create new threads
    programThread = programThread(guiShared)
    # Start new Threads
    programThread.start()

    #start the gui:
    gui.start_gui(guiShared)

    print ("Exiting Main Thread")

