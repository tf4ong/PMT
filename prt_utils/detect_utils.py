import time
import os
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import pandas as pd
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tqdm import tqdm
import ffmpy


def convert2mp4(vid):
    #vid=vid_path +'/raw.h264'
    ff = ffmpy.FFmpeg(inputs={f'{vid}': f' -i {vid}'}, outputs={f'{vid_path}/raw.mp4':'-vcodec copy'})
    try:
        ff.run()
        print('mp4 file generated')
    except Exception as e:
        print(e)
        print('file may have existed already,please check folder path')
    return
    
def remove_overinter(x):
    first_idx=x.first_valid_index()
    last_idx=x.last_valid_index()
    x2=x.loc[first_idx:last_idx]
    return x2
def consecutive_motion_detection(path,gap_fill=2,frame_thres=15):
    df1=pd.read_csv(path+'/yolo_dets.csv')
    temp=[1 if i =='Motion' else np.nan for i in df1.motion.values]
    temp=pd.Series(temp)
    temp=temp.interpolate(method ='linear',limit_direction ='both', limit = gap_fill)
    temp=np.split(temp, np.where(np.isnan(temp))[0])
    temp=[t.drop(t.index[0]) for t in temp if len(t)>frame_thres]
    inds=[i.index.tolist() for i in temp]
    ind=[i for sublists in inds for i in sublists ]
    values=['Motion' if i in ind else 'No_motion' for i in range(len(df1))]
    df1['motion']=values
    df1.to_csv(path+'/yolo_dets.csv')
    return df1
def pure_motion_detect(folder,blur_filter_k_size,motion_area_thresh,
                  intensity_thres,motion_interpolation,len_motion_thres,write_vid=False):
#https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
#Gaussian Mixture Model-based foreground and background segmentation:
    videopath = folder+'/raw.avi'
    vid = cv2.VideoCapture(videopath)
    pbar = tqdm(total=int(vid.get(cv2.CAP_PROP_FRAME_COUNT)),position=0,leave=True)
    with open(folder+'/motion.csv','w') as file:
        file.write('frame,motion,motion_roi\n')
    if write_vid:
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(folder+'/yolov4_det.avi', codec, fps, (width, height))
    avg=None
    frame_count=0
    while vid.isOpened:
        return_value, frame = vid.read()
        if return_value:  
            avg, status,cnt_bb= utils.motion_detection(frame,avg,k_size=(blur_filter_k_size,blur_filter_k_size), 
                                                       min_area=motion_area_thresh,intensity_thres=intensity_thres)
            with open(folder+'/motion.csv','a') as file:
                file.write(f'{str(frame_count)},{status},"{cnt_bb}"\n')
            frame_count+=1
            pbar.update(1)
        else:
            vid.release()
            break
        
        
        
#consecutive_motion_detection(folder,gap_fill=motion_interpolation,frame_thres=len_motion_thres)
    
def yolov4_detect_vid(folder,config_dic_detect,write_vid=False):
    size=config_dic_detect['size']
    weightspath=config_dic_detect['weightpath'][0]
    iou=config_dic_detect['iou']
    score=config_dic_detect['score']
    blur_filter_k_size=config_dic_detect['blur_filter_k_size']
    motion_area_thresh=config_dic_detect['motion_area_thresh']
    intensity_thres=config_dic_detect['motion_area_thresh']
    motion_interpolation=config_dic_detect['motion_interpolation']
    len_motion_thres=config_dic_detect['len_motion_thres']
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(config_dic_detect['model'],config_dic_detect['classes'])
    input_size = size
    videopath = folder+'/raw.avi'
    if not os.path.exists(videopath):
        videopath = folder+'/raw.mp4'
    print(f'Starting to yolov4 and motion detection process on  {videopath}')
    if config_dic_detect['framework'] == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(weightspath, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
    vid = cv2.VideoCapture(videopath)
    pbar = tqdm(total=int(vid.get(cv2.CAP_PROP_FRAME_COUNT)),position=0,leave=True)
    with open(folder+'/yolo_dets.csv','w') as file:
        file.write('frame,bboxes,motion,motion_roi\n')
    if write_vid:
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(folder+'/yolov4_det.avi', codec, fps, (width, height))
    avg=None
    frame_count=0
    while vid.isOpened:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            if config_dic_detect['framework'] == 'tflite':
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))  
            else:
                batch_data = tf.constant(image_data)
                pred_bbox = infer(batch_data)
                #print(pred_bbox)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]
            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=iou,
                score_threshold=score
                )
            pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
            image = utils.draw_bbox(frame,pred_bbox,config_dic_detect['classes'])
            avg, status,cnt_bb= utils.motion_detection(frame,avg,k_size=(blur_filter_k_size,blur_filter_k_size),
                                                       min_area=motion_area_thresh,intensity_thres=intensity_thres)
            with open(folder+'/yolo_dets.csv','a') as file:
                file.write(f'{str(frame_count)},"{image[1]}",{status},"{cnt_bb}"\n')
            frame_count+=1
            #print(image[1])
            pbar.update(1)
        else:
            vid.release()
            break
    consecutive_motion_detection(folder,gap_fill=motion_interpolation,frame_thres=len_motion_thres)
    print(f'Mice Detection and motion detection complete for folder {folder}')
    print(f'Results saved in {folder+"/yolo_dets.csv"}')
    if write_vid: 
        columns=['frame', 'bboxes', 'motion_roi']
        dics={i: eval for i in columns}
        df_dets=pd.read_csv(folder+'/'+'yolo_dets.csv',converters=dics,index_col=False)
        vid = cv2.VideoCapture(videopath)
        pbar = tqdm(total=int(vid.get(cv2.CAP_PROP_FRAME_COUNT)),position=0,leave=True)
        frame_count=0
        print('Starting to write video')
        while vid.isOpened():
            return_value, frame = vid.read()
            if return_value:
                status=df_dets.iloc[frame_count]['motion']
                motion_rois=df_dets.iloc[frame_count]['motion_roi']
                detections=df_dets.iloc[frame_count]['bboxes']
                for objects in detections:
                    xmin, ymin, xmax, ymax = int(objects[0]), int(objects[1]), int(objects[2]), int(objects[3])
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 3)
                    cv2.putText(frame, 'Rodent', (xmin+15, ymin+15), 0, 5e-3 * 200, (0,255,0), 3)
                if status =='Motion':
                    #overlay=blankimg.copy()
                    for c in motion_rois:
                        xmin, ymin, xmax, ymax = int(c[0]), int(c[1]),int(c[2]), int(c[3])
                        sub_img = frame[ymin:ymax, xmin:xmax]
                        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
                        res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
                        frame[ymin:ymax, xmin:xmax] = res
                font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                font_color = (255, 255, 255)
                thick = 1
                text = f"Frame: {str(frame_count)}, Motion:{status}"
                font_size = 0.9
                (text_width, text_height) = cv2.getTextSize(text, font, font_size, thick)[0]
                text_height += 15
                mask = np.zeros((text_height, text_width), dtype=np.uint8)
                mask = cv2.putText(mask,text,(0,15),font,font_size,font_color,thick,cv2.LINE_AA)
                mask = cv2.resize(mask, (frame.shape[1], text_height))
                mask = cv2.merge((mask, mask, mask))
                frame[-text_height:, :, :] = cv2.bitwise_or(frame[-text_height:, :, :], mask)
                #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
                frame_count+=1
                pbar.update(1)
            else:
                print('Finished Writing video; Check for detection and motion detecion accurarcy')
                print('Adjust Parameters as needed')
                out.release()
                break
def yolov4_detect_images(img_list,config_dic_detect,save_folder,save_out=True):
    size=config_dic_detect['size']
    weightspath=config_dic_detect['weightpath'][0]
    iou=config_dic_detect['iou']
    score=config_dic_detect['score']
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(config_dic_detect['model'],config_dic_detect['classes'])
    input_size = size
    saved_model_loaded = tf.saved_model.load(weightspath, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']
    prediction_results = {}
    for img in img_list:
        frame = cv2.imread(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        #print(pred_bbox)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
            )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image = utils.draw_bbox(frame,pred_bbox,config_dic_detect['classes'])
        prediction_results[img] = image[1]
    with open(save_folder+'/predicts.csv','w') as file:
        file.write('img,predicts\n')
    with open(save_folder+'/predicts.csv','a') as file: 
        for i,v in prediction_results.items():
            file.write(f'{i},"{v}"\n')
    #if save_out:
    #    df = pd.DataFrame.from_dict(prediction_results)
    #    df = df.reset_index()
    #    df.to_csv(f'{save_folder}/pred_results.csv')
    return prediction_results 