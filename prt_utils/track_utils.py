import numpy as np
import math
import cv2
from numba import jit
from itertools import islice



"""
simple functions 
"""
def get_tags(path):
    with open(path+'/'+'logs.txt','r') as f:
        tags=f.readlines()
        tags=[int(i) for i in tags[1][6:].split(',')]
    return tags

def get_ids(x):
    ids=[]
    if x!=[]:
        ids=[i[4] for i in x]
    return ids


def get_unique_ids(x):
    newlist=[]
    for i in x:
        if len(i)!=0:
            for y in i:
                newlist.append(y)
    else:
        pass
    newlist=list(set(newlist))
    return newlist

@jit
def bbox_area(bbox):
    '''
    calculates the area of the bbself.df_track_temp
    Intake bb:x1,y1,x2,y2
    ''' 
    w=abs(bbox[0]-bbox[2])
    h=abs(bbox[1]-bbox[3])
    return w*h

def float_int(x):
    rounded=[]
    for i in x:
        tracks=[int(f) for f in i]
        #tracks.append(int(i[4]))
        rounded.append(tracks)
    return rounded

def bbox_to_centroid(bbox):
    '''
    returns the centroid of the bbox
    '''
    if bbox!=[]:
        cX=(bbox[0]+bbox[2])/2
        cY=(bbox[1]+bbox[3])/2
        return [int(cX),int(cY)]
    else:
        return []


def Distance(centroid1,centroid2):
    ''' 
    calculates the centronoid distances between bbs
    intake centronoid
    '''
    dist = math.sqrt((centroid2[0] - centroid1[0])**2 + (centroid2[1] - centroid1[1])**2)
    return dist

@jit
def iou(bb_test,bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
      + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return(o)

def sublist_decompose(list_of_list):
    return [bpt for bpts in list_of_list for bpt in bpts]


def bboxContains(bbox,pt,slack=0.05):
    bb=apply_slack_bb(bbox,slack)
    logic = bb[0]-slack<= pt[0] <= bb[2]+slack and bb[1]-slack<= pt[1] <= bb[3]+slack
    return logic

def RFID_ious(RFID, bbox,RFID_coords):
    reader_coords=RFID_coords[int(RFID)]
    iou_reader=iou(reader_coords,bbox)
    return iou_reader

def reconnect_id_update(reconnect_ids,id2remove):
    for i in id2remove:
        del reconnect_ids[i]
    return reconnect_ids

def distance_box_RFID(RFID,bbox,RFID_coords):
    '''
    Gets the centroid distance between RFID reader of interest and bbox
    '''
    bbox_1_centroid=bbox_to_centroid(RFID_coords[int(RFID)])
    bbox_2_centroid=bbox_to_centroid(bbox)
    return Distance(bbox_1_centroid,bbox_2_centroid)

'''
Gets the distance of the bb to RFIDs
'''
def distance_to_entrance(bbox2,RFID_coords,entrance_reader):
    bbox_1_centroid=bbox_to_centroid(RFID_coords[entrance_reader])
    bbox_2_centroid=bbox_to_centroid(bbox2)
    return Distance(bbox_1_centroid,bbox_2_centroid)

def list_split(it,size):
    it = iter(it)
    return list(iter(lambda: tuple(islice(it, size)), ()))


def apply_slack_bb(bb,slack):
    x_slack=0.5*slack*(bb[2]-bb[0])
    y_slack=0.5*slack*(bb[3]-bb[1])
    bbox=[bb[0]+int(x_slack),bb[1]+int(y_slack),bb[2]+int(x_slack),bb[3]+int(y_slack),bb[4]]
    return bbox

def apply_slack(listbb,slack):
    return [apply_slack_bb(bb, slack) for bb in listbb]

def bb_contain_mice_check(frame,bbox,diff_bg):
    xstart,ystart,xend,yend= int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cropped_bbox=frame[ystart:yend,xstart:xend]
    try:
        if np.absolute(np.mean(frame)-np.mean(cropped_bbox))> diff_bg:
            return True
        else:
            return False
    except Exception:
        return True

def array2list(array):
    return [i.tolist() for i in array]

def roatate_frame(frame, degree):
    (h, w) = frame.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # rotate our image by 45 degrees around the center of the image
    M = cv2.getRotationMatrix2D((cX, cY), degree, 1.0)
    rotated = cv2.warpAffine(frame, M, (w, h))
    return rotated


    
