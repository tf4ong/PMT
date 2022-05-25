import cv2
import pandas as pd
import numpy as np
from psyco_utils.track_utils import *
from absl import app, flags, logging
from absl.flags import FLAGS
from tqdm import tqdm 
import os


flags.DEFINE_string('vid_path',None,'path to video and RFID csv file for tracker cage')

color_dic={'blue':(255,0,0),'purple':(128,0,128),
           'orange':(155,140,0),'yellow':(0,255,255),'aqura':(212,255,127),
           'magenta':(255,0,255),'lbrown':(181,228,255),'green':(0,255,0)}

def read_match_frames(path):
    c=['reader_coord', 'sort_tracks', 'Dist_r','iou_r']
    c={i: eval for i in c}
    df=pd.read_csv(path+'/matching_process_cage.csv',index_col=False,converters=c)
    return df

def get_frame(path,frame_count):
    cap=cv2.VideoCapture(path+'/raw.avi')
    cap.set(1,frame_count)
    ret, frame = cap.read()
    cap.release()
    return frame

def edit_image(stracks,distances,reader,ious,frame_n,path):
    frame_test=get_frame(path,frame_n)
    cv2.rectangle(frame_test,(reader[0], reader[1]), (reader[2], reader[3]), (0,0,0), 7)
    cent_point_reader=bbox_to_centroid(reader)
    cv2.circle(frame_test,(cent_point_reader[0],cent_point_reader[1]),10,(0,0,0),-1)
    c_count=0
    id_color={}
    for objects in stracks:
        color=list(color_dic.keys())[c_count]
        id_color[objects[4]]=color
        cent_point=bbox_to_centroid(objects)
        xmin, ymin, xmax, ymax, index = int(objects[0]), int(objects[1]),\
            int(objects[2]), int(objects[3]), int(objects[4])
        cv2.rectangle(frame_test, (xmin, ymin), (xmax, ymax), color_dic[color], 3)    
        cv2.circle(frame_test,(cent_point[0],cent_point[1]),10,color_dic[color],-1)
        cv2.line(frame_test,(cent_point[0],cent_point[1]),(cent_point_reader[0],cent_point_reader[1]),
                 color_dic[color],thickness=3)
        c_count+=1
    height=frame_test.shape[0]
    width=frame_test.shape[1]
    blank_img=255*np.ones(shape=[height+100,width+250,3],dtype=np.uint8)
    blank_img[50:height+50,0:width]=frame_test
    cv2.putText(blank_img,f"Frame: {str(frame_n)}",(int(width/2)-120,30),0, 5e-3 * 200,(0,0,255),3)
    cv2.putText(blank_img,'Distance to Reader', (width+20, 25), 0, 5e-3 * 120, (0,0,0), 2)
    spacer=0
    for line,dist in zip(id_color.values(),distances):
        cv2.line(blank_img,(int(width+50),int(75+spacer)),(int(width+125),int(75+spacer)),color_dic[line], thickness=5, lineType=4)
        cv2.putText(blank_img,': '+str(round(dist)), (int(width+150), int(82+spacer)), 0, 5e-3 * 150, (0,0,0), 2)
        spacer+=25
    spacer+=50
    cv2.putText(blank_img,'IOU with Reader', (width+20, 75+spacer), 0, 5e-3 * 120, (0,0,0), 2)
    spacer+=50
    for line,iou in zip(id_color.values(),ious): 
        cv2.line(blank_img,(int(width+50),int(75+spacer)),(int(width+125),int(75+spacer)),color_dic[line], thickness=5, lineType=4)
        cv2.putText(blank_img,': '+str(round(iou,2)), (int(width+150), int(82+spacer)), 0, 5e-3 * 150, (0,0,0), 2)
        spacer+=25
    return blank_img

def save_match_imgs(path):
    df=read_match_frames(path)
    if not os.path.exists(path+'/match_frames'):
        os.mkdir(path+'/match_frames')
    pbar=tqdm(total=len(df),position=0, leave=True)
    for inde in df.index:
        reader=df.iloc[inde]['reader_coord']
        frame_n=df.iloc[inde]['frame']
        stracks=df.iloc[inde]['sort_tracks']
        ious=df.iloc[inde]['iou_r']
        distances=df.iloc[inde]['Dist_r']
        img=edit_image(stracks,distances,reader,ious,frame_n,path)
        cv2.imwrite(path+f'/match_frames/match_frame_{inde}.png',img)
        pbar.update(1)
    return

def main(_argv):
    if FLAGS.vid_path ==None:
        print('Please enter valid path to video folder')
    elif not os.path.exists(FLAGS.vid_path+'/matching_process_cage.csv'):
        print('No matchign file found \nplease confirm folder path and run PSYCO tracking process at least once')
    else:
        save_match_imgs(FLAGS.vid_path)
    return



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass