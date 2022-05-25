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
    return [class_detec,[l,t,r,b]]

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

# create folder to hold results of ground truth, and evaulations 
# load yolo ground truth (done)
# perform detection on image



path = '/home/tony/white_mouse_test/train/label_imgs'


img_coords = [path+'/'+i for i in os.listdir(path) if i[-4:] == '.txt' and \
              i != 'classes.txt']

ground_truth_dic = {}
#a for loop to operate on each image
for gt in img_coords:
    f=open(gt,'r')
    img=cv2.imread(gt[:-4]+'.png')
    dih,diw, _ = img.shape
    lines = f.readlines()
    lines = [i.split() for i in lines]
    ground_truth = [yolo2cvbb(i, dih, diw) for i in lines]
    ground_truth_dic[gt]=ground_truth


a='/media/user/Source/Data/coco_dataset/coco/images/val2017/000000338304.jpg 168,413,243,517,18 65,473,235,640,18 191,422,335,598,18 88,387,207,488,18 266,217,410,466,0 250,237,318,364,0 150,252,212,389,0 94,291,161,425,0 0,298,101,488,0 49,305,123,473,0 200,255,249,360,0 189,0,425,439,0 304,189,384,273,0 241,355,353,486,18 0,483,146,639,18 199,525,347,638,18 36,311,56,339,0 100,286,114,310,0 74,393,122,427,26 33,430,84,496,26 73,265,88,327,0 270,355,365,452,18 193,354,338,494,18 230,155,255,186,67 230,253,279,360,0 0,89,387,591,0\n'
annotation = a.strip().split()
image_path = annotation[0]
image_name = image_path.split('/')[-1]
bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])
bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]






np.array([list(map(int, box.split(','))) for box in a])







a=[{'class':0,'bb':'12 23 36 89','used':False},{'class':0,'bb':'18 200 36 89','used':False}]


b=json.load(open('dumper.json'))