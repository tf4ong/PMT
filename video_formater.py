import time
import os
import time
import cv2
import shutil

def convert2pmt(vid):
    dir_path = os.path.dirname(vid)
    basename = os.path.basename(vid)
    newfold = dir_path+basename[:-4]
    os.mkdir(newfold)
    shutil.move(vid, newfold+'/')
    with open(newfold+'/logs.txt','w') as f:
        f.writelines('mice:1\n')
        f.writelines('tags: 123')
    video =[newfold+'/'+z for z in os.listdir(newfold)][0]
    os.rename(video,newfold+'/raw.mp4')
    vid=cv2.VideoCapture(newfold+'/raw.mp4')
    vid_length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    current_time = time()
    frame_rate=40
    frame_constant = 1/frame_rate
    current_pts = 0
    frame_count = 0 
    with open(newfold+'/timestamps.csv','w') as f:
        f.writelines(f'StartTime: {time()}\n')
        f.writelines('frame,timestamp\n')
        for i in range(vid_length):
            f.writelines(f'{frame_count},{current_pts}\n')
            current_pts+=frame_constant
            frame_count+=1
    vid.release()
    


