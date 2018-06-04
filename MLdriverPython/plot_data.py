import matplotlib.pyplot as plt
import numpy as np
import data_manager1
from hyper_parameters import HyperParameters


HP = HyperParameters()
dm = data_manager1.DataManager(HP.save_file_path,HP.restore_file_path,HP.restore_flag)
dm.plot_all()