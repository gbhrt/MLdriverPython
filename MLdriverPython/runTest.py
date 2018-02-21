from planner import Planner 
import numpy as np
from library import *


if __name__ == "__main__": 
    pl = Planner()
    #pl.simulator.set_address("10.0.17.74",5007)# if run in linux # "10.0.17.74"  "10.2.1.111"
    pl.start_simple()

    #pl.create_path_in_run(0,"path_1.txt")
    pl.read_path_data("path_1.txt")
    pl.desired_path = pl.path_to_global(pl.desired_path)
    pl.simulator.send_path(pl.desired_path)
    pl.run_on_path()

    pl.end()
    print("done")

