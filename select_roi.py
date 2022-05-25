import cv2
#import os
#import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
#from sklearn.cluster import KMeans

flags.DEFINE_string('vid_path', './data/coords_example', 'path to PSYCO video folder')
flags.DEFINE_integer('reader_count', 9, 'Number of RFID readers in the video')
flags.DEFINE_integer('frame_count', 1, 'The frame to mark the RFID readers')


def main(_argv):
    cap=cv2.VideoCapture(FLAGS.vid_path+'/raw.avi')
    print(FLAGS.vid_path)
    cap.set(1,FLAGS.frame_count)
    ret, frame = cap.read()
    count=FLAGS.reader_count
    n_select=0
    RFID_coords={}
    while n_select<count:
        x,y,w,h=cv2.selectROI(frame)
        print(x,y,x+w,y+h)
        #cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,25))
        #roi_cropped=frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        coord_n=input('enter reader number')
        RFID_coords[int(coord_n)]=[x,y,x+w,y+h]
        n_select+=1
    print(RFID_coords)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
