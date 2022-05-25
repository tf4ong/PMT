# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:23:42 2022

@author: Tony
"""

import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
def read_label_map(label_map_path):

    item_id = None
    item_name = None
    items = {}

    with open(label_map_path, "r") as file:
        for line in file:
            line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif "id" in line:
                item_id = int(line.split(":", 1)[1].strip())
            elif "name" in line:
                #print(line)
                #print(line.split(" ")[-1][1:-2])
                #item_name = line.split(" ")[-1].replace("\"", " ").strip()
                item_name = line.split(" ")[-1][1:-2]
            if item_id is not None and item_name is not None:
                items[item_name] = item_id
                item_id = None
                item_name = None

    return items

def read_pvoc(data_path):
    img_inds_file = os.path.join(data_path, 'ImageSets', 'Main/')
    files= [img_inds_file + i for i in os.listdir(img_inds_file)]
    with open(files[0]) as fp:
        data = fp.read()
    if len(files)>1:
        data=data+'\n'
        for i in files[1:]:
            with open(i) as fp:
                data += fp.read()
    imge_inds = [i for i in data.split('\n')]
    imge_inds = [i.split('.png')[0] for i in imge_inds]
    return list(dict.fromkeys(imge_inds))

def get_labled_imgs(data_path):
    return [i for i in os.listdir(data_path+'JPEGImages')]

def parse_text(ims,data_path,directory,filename,use_difficult_bbox= True):
    a=read_label_map(data_path+'pascal_label_map.pbtxt')
    pbar = tqdm(total=len(ims),leave=True,position=0)
    with open(f'{directory}/{filename}.txt', 'a') as f:
        for img in ims:
            image_path = os.path.join(data_path, 'JPEGImages', img )
            annotation = image_path
            label_path = os.path.join(data_path, 'Annotations', img[:-4]+ '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            #print(annotation)
            #import sys; sys.exit()
            for obj in objects:
                difficult = obj.find('difficult').text.strip()
                if (not use_difficult_bbox) and(int(difficult) == 1):
                    continue
                bbox = obj.find('bndbox')
                class_ind=a[obj.find('name').text.lower()]
                #class_ind = classes.index(obj.find('name').text.lower().strip())
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
            f.write(annotation + "\n")
            #print(annotation)
            pbar.update(1)
    return f'{directory}/{filename}.txt'


'''
path = 'D:/tracker_test/labelled_out/mouse-PascalVOC-export/pascal_label_map.pbtxt'
#print(read_label_map(path))
a=read_label_map(path)
look = 'train'
data_path = 'D:/tracker_test/labelled_out/mouse-PascalVOC-export/'
train_fraction = 0.8
#data=read_pvoc(data_path)
ims= get_labled_imgs(data_path)
train=int(len(ims)*train_fraction) 
train_ims = ims[:train]
test_ims = ims[train:]
parse_text(train_ims, data_path,data_path,'train',use_difficult_bbox= True)
parse_text(test_ims, data_path,data_path,'test',use_difficult_bbox= True)
'''






