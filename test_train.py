from PyRodentTracks import PRT_train
temp=PRT_train('/home/tony/white_mouse_test/train',False)
pvoc_path='/home/tony/white_mouse_test/mouse-PascalVOC-export/'
temp.load_config()
#temp.pvocTrainImport(pvoc_path)
#temp.train_yolo_network()
#from PyRodentTracks import PRT_analysis
#path = '/home/tony/white_mouse_test/vid_test/bh_hc/'
#config_path = '/home/tony/white_mouse_test/train/config.ini'
#a= PRT_analysis(path,config_path)
#a.detect_mice()