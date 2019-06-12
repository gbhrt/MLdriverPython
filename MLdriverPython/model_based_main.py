#import gui
import tkinker_gui
import threading
import time
import model_based_run
import shared
import hyper_parameters
import data_manager1

#python runsnake.py myscript.profile
#snakeviz myscript.profile
#snakeviz action_prof.profile
#snakeviz train_prof.profile
class programThread (threading.Thread):
   def __init__(self,guiShared,HP,dataManager):
        threading.Thread.__init__(self)
        self.guiShared = guiShared
        self.HP = HP
        self.dataManager = dataManager
      
   def run(self):
        print ("Starting " + self.name)
        model_based_run.run(self.guiShared,self.HP,self.dataManager)
        print ("Exiting " + self.name)


if __name__ == "__main__": 

    guiShared = shared.guiShared()
    HP = hyper_parameters.ModelBasedHyperParameters()
    dataManager = data_manager1.DataManager(HP.save_file_path,HP.restore_file_path,HP.restore_flag)

    # Create new thread
    programThread = programThread(guiShared,HP,dataManager)
    programThread.start()

    if HP.gui_flag:
        #start the gui:
        tkinker_gui.TkGui(guiShared)
    while(not guiShared.exit):
        time.sleep(1)
        continue

    

    print ("Exiting Main Thread")

