import os




f= open('/home/tony/white_mouse_test/train/trainning/train.txt','r')



def get_ind(filename,char):
    return [i for i,v in enumerate(filename) if v == char]




'''
path='/media/tony/data/data/tracker_trainningsets/spt_tracker-PascalVOC-export/JPEGImages/'

files2rename=[path + i for i in os.listdir(path) if len(get_ind(i,'.'))>1]

errs=[]
for i in files2rename:
    ind = get_ind(os.path.basename(i),'.')
    if len(ind)>2:
        errs.append(i)
    else:
        bn=os.path.basename(i)[:ind[1]]
        final_name = os.path.split(files2rename[0])[0]+'/'+bn
        os.rename(i, final_name)
'''

'''
path='/media/tony/data/data/tracker_trainningsets/spt_tracker-PascalVOC-export/ImageSets/Main/'

files=[path +i for i in os.listdir(path)]


err=[]
file= open(files[1],'r')
lines=file.readlines()
for i in range(len(lines)):
    frags=lines[i].split(' ')
    if len(get_ind(frags[0],'.'))>1:
        frags[0] = frags[0][:get_ind(frags[0],'.')[1]]
        lines[i]=' '.join(frags)
    elif len(get_ind(frags[0],'.'))>2:
        err.append(i)
a_file=open(files[1],'w')
a_file.writelines(lines)
a_file.close()
'''

import xml.etree.ElementTree as ET
from tqdm import tqdm 

path='/media/tony/data/data/tracker_trainningsets/spt_tracker-PascalVOC-export/Annotations/'
files=[path+i for i in os.listdir(path)]
pbar=tqdm(total=len(files),leave=True,position=0)
#x='/mnt/team/TM_Lab/Tony/tracker_cage/yolo_trainning_set/spt_tracker-PascalVOC-export/abc_frame2322.jpg.xml'

for f in files:
    tree = ET.parse(f)
    root = tree.getroot()
    for i in root.iter('filename'):
        inds=get_ind(i.text,'.')
        if len(inds)>1:
            i.text = i.text[:inds[1]]
    for i in root.iter('path'):
        bn=os.path.split(i.text)[1]
        pname=os.path.split(i.text)[0]
        inds=get_ind(bn,'.')
        if len(inds)>1:
            bn = bn[:inds[1]]
            i.text=pname+'/'+bn
    tree.write(f)
    pbar.update(1)


path='/media/tony/data/data/tracker_trainningsets/spt_tracker-PascalVOC-export/Annotations/'

files2rename=[path + i for i in os.listdir(path) if len(get_ind(i,'.'))>1]

errs=[]
for i in files2rename:
    bnames=os.path.basename(i).split('.')
    bname=bnames[0]+'.'+bnames[2]
    fname=os.path.split(i)[0]+'/'+bname
    os.rename(i, fname)

