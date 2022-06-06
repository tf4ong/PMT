from PyRodentTracks import PRT_train

temp=PRT_train({path2folder},False)
temp.label_imgs()


temp.evaluate_weights()


#config_path = '/home/tony/white_mouse_test/train/config.ini'
#a= PRT_analysis(path,config_path)
#a.detect_mice()