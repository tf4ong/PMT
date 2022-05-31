from configparser import ConfigParser
import configparser


def detect_config_loader(path):
    config = ConfigParser()
    config.read(path)
    cfg = 'PRT_Detection'
    config_dic={}
    config_dic['tiny']=eval(config.get(cfg, 'tiny'))
    config_dic['framework'] = str(config.get(cfg, 'framework'))
    config_dic['model'] = str(config.get(cfg, 'model'))
    config_dic['classes'] = str(config.get(cfg, 'classes'))
    if ',' in config_dic['classes']:
        config_dic['classes'] = config_dic['classes'].split(',')
    else:
        config_dic['classes'] = [config_dic['classes']]
    config_dic['weightpath']=str(config.get(cfg, 'weightpath')).split(',')
    config_dic['score']=float(config.get(cfg, 'score'))
    config_dic['iou']=float(config.get(cfg, 'iou'))
    config_dic['size']=int(config.get(cfg, 'size'))
    config_dic['motion_interpolation']=int(config.get(cfg, 'motion_interpolation'))
    config_dic['motion_area_thresh']=int(config.get(cfg, 'motion_area_thresh'))
    config_dic['len_motion_thres']=int(config.get(cfg, 'len_motion_thres'))
    config_dic['blur_filter_k_size']=int(config.get(cfg, 'blur_filter_k_size'))
    config_dic['intensity_thres']=int(config.get(cfg, 'intensity_thres'))
    return config_dic

def train_config_loader(path):
    config = ConfigParser()
    config.read(path)
    cfg = 'PRT_Train'
    config_dic={}
    config_dic['frames2pic'] = int(config.get(cfg, 'frames2pic'))
    config_dic['size']=int(config.get(cfg, 'size'))
    config_dic['trainfraction'] = float(config.get(cfg,'trainfraction'))
    config_dic['classes'] = str(config.get(cfg, 'classes'))
    if ',' in config_dic['classes']:
        config_dic['classes'] = config_dic['classes'].split(',')
    else:
        config_dic['classes'] = [config_dic['classes']]
    config_dic['classes'] = {i:z for i,z in zip(range(len(config_dic['classes'])),config_dic['classes'])}
    config_dic['batch'] = int(config.get(cfg, 'batch'))
    config_dic['random'] = eval(config.get(cfg, 'random'))
    config_dic['max_batches'] = int(config.get(cfg, 'max_batches'))
    config_dic['filters'] = int(config.get(cfg, 'filters'))
    config_dic['steps'] = list(map(int,(config.get(cfg, 'steps')).split(',')))
    config_dic['subdivision']=int(config.get(cfg, 'subdivision'))
    return config_dic
    
def tracking_config_loader(path):
    config = ConfigParser()
    config.read(path)
    cfg = 'Tracking'
    config_dic={}
    config_dic['max_age']=int(config.get(cfg, 'max_age'))
    config_dic['min_hits']=int(config.get(cfg, 'min_hits'))
    config_dic['iou_threshold']= float(config.get(cfg, 'iou_threshold'))
    config_dic['interaction_thres']= float(config.get(cfg, 'interaction_thres'))
    config_dic['iou_min_sbb_checker']= float(config.get(cfg, 'iou_min_sbb_checker'))
    config_dic['sbb_frame_thres']= int(config.get(cfg, 'sbb_frame_thres'))
    config_dic['leap_distance']= int(config.get(cfg, 'sbb_frame_thres'))
    config_dic['resolution']=list(map(int, config.get(cfg, 'resolution').split(',')))
    return config_dic


def analysis_config_loader(path):
    config = ConfigParser()
    config.read(path)
    cfg = 'PRT_RFID_Matching'
    config_dic={}
    config_dic['RFID_readers']=eval(str(config.get(cfg, 'RFID_readers')))
    config_dic['entrance_time_thres']=float(config.get(cfg, 'entrance_time_thresh'))
    config_dic['entrance_distance']=float(config.get(cfg, 'entrance_distance'))
    config_dic['correct_iou']= float(config.get(cfg, 'correct_iou'))
    config_dic['RFID_dist']= float(config.get(cfg, 'RFID_dist'))
    config_dic['entr_frames']=int(config.get(cfg, 'entr_frames'))
    config_dic['reader_thres']= float(config.get(cfg, 'reader_thres'))
    config_dic['trac_interpolation']=int(config.get(cfg, 'trac_interpolation'))
    config_dic['itc_slack']=float(config.get(cfg, 'itc_slack'))
    entrance_readers= str(config.get(cfg, 'entrance_reader')).split(',')
    entrance_readers[0]=eval(entrance_readers[0])
    entrance_readers[1]=int(entrance_readers[1])
    config_dic['entrance_reader']=entrance_readers
    if config_dic['entrance_reader'][0]: 
        config_dic['entrance_reader']=config_dic['entrance_reader'][1]
    else: 
        config_dic['entrance_reader']=None
    return config_dic

def dlc_config_loader(path):
    config = ConfigParser()
    config.read(path)
    cfg = 'DLC'
    config_dic={}
    config_dic['dbpt']=str(config.get(cfg, 'dbpts')).split(',')
    config_dic['dbpt_distance_compute']=str(config.get(cfg, 'dbpt_distance_compute')).split(',')
    config_dic['dbpt_int']=str(config.get(cfg, 'dbpt_int')).split(',')
    config_dic['dbpt_box_slack']=float(config.get(cfg, 'dbpt_box_slack'))
    return config_dic


def set_value(file_path, section, key, value):
    config = configparser.RawConfigParser()
    config.read(file_path)
    config.set(section,key,value)                         
    cfgfile = open(file_path,'w')
    config.write(cfgfile, space_around_delimiters=False)  # use flag in case case you need to avoid white space.
    cfgfile.close()