import numpy as np
import os
import matplotlib.pylab as plt
import cv2

# conver yolo bb to cv2 bounding boxes (xmin,ymin, xmax,ymax)
# Taken from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
# via https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection#comment102178409_44592380
def yolo2cvbb(yolobb,dih,diw):
    class_detec = int(yolobb[0])
    bbox = yolobb[1:]
    bbox = [float(i) for i in bbox]
    x, y, w, h = map(float, bbox)
    l = int((x - w / 2) * diw)
    r = int((x + w / 2) * diw)
    t = int((y - h / 2) * dih)
    b = int((y + h / 2) * dih)
    return [l,t,r,b,1,class_detec]

def read_gt(path2gt):
    img_coords = [path2gt+'/'+i for i in os.listdir(path2gt) if i[-4:] == '.txt' and \
                  i != 'classes.txt']
    if len(img_coords) ==0:
        print('No labels detected, please confirm that images are labeled')
        return
    else:
        ground_truth_dic = {}
        for gt in img_coords:
            f=open(gt,'r')
            img=cv2.imread(gt[:-4]+'.png')
            dih,diw, _ = img.shape
            lines = f.readlines()
            lines = [i.split() for i in lines]
            ground_truth = [yolo2cvbb(i, dih, diw) for i in lines]
            ground_truth_dic[gt]=ground_truth
        return ground_truth_dic
