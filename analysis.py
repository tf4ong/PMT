import os
import pandas as pd
import numpy as np
#from tensorflow.python.client import device_lib
import PyRodentTracks as prt
#print(device_lib.list_local_devices())

rfid_tags=[34443625031,141868466231,34443624660]
config_path = '/media/tony/data/data/large_tracker_trainning/config.ini'


path='/mnt/team/TM_Lab/Tony/stroke_3d_cylinder/tracker_cage/BJ/'

folders=['2022-08-19_16-13-19','2022-08-20_16-48-44','2022-08-21_16-22-57','2022-08-22_16-34-54','2022-08-23_15-41-55']

for i in folders:
    temp = prt.PRT_analysis(path+i,config_path,3,rfid_tags)
    #temp.detect_mice()
    temp.load_RFID()
    temp.load_dets()
    temp.RFID_match()
    temp.find_activte_mice()
    temp.compile_travel_trjectories()
