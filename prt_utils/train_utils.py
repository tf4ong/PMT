import os
from random import shuffle
import shutil


def set_text(img_list,ty, dest):
    with open(f"{dest}{ty}.txt",'w') as outfile:
        for img in img_list:
            if ty == 'train':
                outfile.write(f'trainning/train_set/{img}\n')
            elif ty == 'test':
                outfile.write(f'trainning/test_set/{img}\n')
            else:
                print(f'{ty} not train or test')
        outfile.close()

def generate_sets(vid_fold,train_frac,dest):
    img_list = [img for img in os.listdir(vid_fold) if img[-4:]=='.png' or img[-4:]=='.jpg']
    shuffle(img_list)
    train=int(len(img_list)*train_frac) 
    train_ims = img_list[:train]
    test_ims = img_list[train:]
    os.mkdir(f'{dest}train_set')
    os.mkdir(f'{dest}test_set')
    set_text(train_ims,'train', dest)
    set_text(test_ims,'test',dest)
    for i in train_ims:
        textfile = f'{i[:-4]}.txt'
        shutil.copyfile(f'{vid_fold}/{i}', f'{dest}train_set/{i}')
        try:
            shutil.copyfile(f'{vid_fold}/{textfile }', f'{dest}train_set/{textfile }')
        except Exception as e:
            open(f'{dest}train_set/{i[:-4]}.txt','w').close()
            #shutil.copyfile(f'{vid_fold}/{i}', f'{dest}train_set/{i}')
            continue 
    for i in test_ims:
        textfile = f'{i[:-4]}.txt'
        shutil.copyfile(f'{vid_fold}/{i}', f'{dest}test_set/{i}')
        try:
            shutil.copyfile(f'{vid_fold}/{textfile }', f'{dest}test_set/{textfile }')
        except Exception as e:
            open(f'{dest}test_set/{i[:-4]}.txt','w').close()
            #shutil.copyfile(f'{vid_fold}/{textfile }', f'{dest}test_set/{textfile }')
            continue


def generate_pathtext(path,nclass):
    if not os.path.exists(f'{path}checkpoints'):
        os.mkdir(f'{path}checkpoints')
    with open(f'{path}obj.data','w') as out:
        out.write(f'classes = {nclass}\n')
        out.write('train = trainning/train.txt\n')
        out.write('test = trainning/test.txt\n')
        out.write('names = trainning/obj.names\n')
        out.write('backup = trainning/checkpoints')
    return

def load_train_settings(train_config_dict,dest):
    batch= train_config_dict['batch']
    width= train_config_dict['size']
    height= train_config_dict['size']
    random = train_config_dict['random']
    max_batches = train_config_dict['max_batches']
    filters= train_config_dict['filters']
    classes = len(train_config_dict['classes'])
    steps = train_config_dict['steps']
    subdivision = train_config_dict['subdivision']
    f= open('./configs/yolov4-obj.cfg','r')
    lines= f.readlines()
    lines[5]= f'batch={batch}\n'
    lines[6]= f'subdivisions={subdivision}\n'
    lines[7]= f'width={width}\n'
    lines[8]= f'height={height}\n'
    lines[21] = f'steps={steps[0]},{steps[1]}\n'
    if random:
        lines[1150]='random=1\n'
    else:
        lines[1150]='random=0\n'
    lines[19]= f'max_batches = {max_batches}\n'
    lines[962]= f'filters={filters}\n'
    lines[1050]= f'filters={filters}\n'
    lines[1138]= f'filters={filters}\n'
    lines[969]= f'classes={classes}\n'
    lines[1057]= f'classes={classes}\n'
    lines[1145]= f'classes={classes}\n'
    with open (f'{dest}yolo4-obj.cfg','w')as out:
        for line in lines:
            out.write(line)
        out.close()
    return