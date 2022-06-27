import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
from prt_utils.track_utils import *
import multiprocessing as mp
import itertools
import os





#need to implement fast video processing in the future
#multi-threading/mulitprocessing?


def generate_RFID_video(path,df_RFID,tags,df_tracks_out,validation_frames,config_dict_analysis,config_dic_dlc,dlc_bpts=False,
                        plot_motion=False,out_folder=None,plot_readers=True):
    frame_count=0
    RFID_coords=config_dict_analysis['RFID_readers']
    entrance_reader=config_dict_analysis['entrance_reader']
    vid=cv2.VideoCapture(path+'/raw.avi')
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if out_folder == None:
        #print(path)
        out = cv2.VideoWriter(path+'/RFID.avi', codec, fps, (width+700, height+200))
    else:
        name=os.path.basename(path)
        out = cv2.VideoWriter(out_folder+'/'+name+'.avi', codec, fps, (width+700, height+200))
    vid_length=len(df_RFID)
    pbar = tqdm(total=vid_length,position=0, leave=True)
    RFID_tracks=df_tracks_out['RFID_tracks'].to_list()
    if plot_motion:
        motion_status=df_tracks_out['motion'].to_list()
        motion_rois=df_tracks_out['motion_roi'].to_list()
    if dlc_bpts:
        bpts=df_tracks_out['bpts'].to_list()
        #bpts_pairs=df_tracks_out['dbpt2look'].to_list()
        if len(tags)>1:
            dlc_columns=[f'{i[0]}_{i[1]}' for i in itertools.combinations(tags,2)]
            dlc_columns=[df_tracks_out[i].to_list() for i in dlc_columns]
            dlc_colors=[[int(n) for n in np.random.choice(range(256), size=3)] for i in range(len(dlc_columns))]
    RFID_readings=[df_tracks_out.iloc[i]['RFID_readings'] for i in validation_frames]
    vid_length=int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    Reader_display={i:z for i,z in zip(validation_frames,RFID_readings)}
    colors=['blue','purple','black','orange','red','yellow','aqura','magenta','lbrown','green']
    tag_codes={tag: num+1 for num, tag in zip(range(len(tags)),tags)}
    tag_codes={i:[tag_codes[i],z] for i,z in zip(tag_codes.keys(),colors)}
    color_dic={'green':(0,255,0),'blue':(255,0,0),'purple':(128,0,128),'black':(0,0,0),
               'orange':(155,140,0),'red':(0,0,255),'yellow':(0,255,255),'aqura':(212,255,127),
               'magenta':(255,0,255),'lbrown':(181,228,255)}
    distances_doc_f=[]
    while vid.isOpened():
        ret,img=vid.read()
        if ret and frame_count<vid_length-1:
            if len([i for i in validation_frames if i>frame_count])>0:
                RFID_frame=[i for i in validation_frames if i>frame_count][0]
            else:
                 RFID_frame=[]
            tracker=RFID_tracks[frame_count]
            blankimg=255*np.ones(shape=[height+200,width+700,3],dtype=np.uint8)
            blankimg[100:height+100,100:width+100]=img
            if tracker !=[]:
                for objects in tracker:
                    xmin, ymin, xmax, ymax, index = int(objects[0]), int(objects[1]),\
                        int(objects[2]), int(objects[3]), int(objects[4])
                    m_display=tag_codes[index][0]
                    m_color=color_dic[tag_codes[index][1]]
                    cv2.rectangle(blankimg, (xmin+100, ymin+100), (xmax+100, ymax+100), m_color, 3)
                    cv2.putText(blankimg, str(m_display), (xmin+90, ymin+95), 0, 5e-3 * 200, m_color,3)
            if plot_motion:
                MS=motion_status[frame_count]
                cv2.putText(blankimg,f"Frame: {str(frame_count)} Motion: {MS}",(int(width/2-width/5),50),0, 5e-3 * 200,(0,0,255),5)
            else:
                cv2.putText(blankimg,f"Frame: {str(frame_count)}",(int(width/2),50),0, 5e-3 * 200,(0,0,255),3)
            spacer=0
            """
            check 
            """
            if RFID_frame != []:
                for i in Reader_display[RFID_frame]:
                    if i[0] == entrance_reader:
                        reader_display='Entrancer'
                    else:
                        reader_display=i[0]
                    m_display=tag_codes[i[1]][0]
                    cv2.putText(blankimg,f'Frame {RFID_frame}: reader {reader_display},tag read {m_display}', \
                                (25,int(1.4*height+spacer)),0,5e-3 * 180,(0,0,255),3)
                    spacer+=10
            if plot_readers:
                for i,v in RFID_coords.items():
                    if i!=entrance_reader:
                        xmin, ymin, xmax, ymax=v[0],v[1],v[2],v[3]
                        cv2.rectangle(blankimg, (xmin+100, ymin+100), (xmax+100, ymax+100), (0,0,0), 2)
                        cent_point=bbox_to_centroid(v)
                        #print(i)
                        #print(cent_point[0])
                        cv2.putText(blankimg,f"{str(i)}",(int(cent_point[0])+100,int(cent_point[1])+100),0, 5e-3 * 200,(0,0,0),2)
            if plot_motion:
                if MS =='Motion':
                    for c in motion_rois[frame_count]:
                        xmin, ymin, xmax, ymax = int(c[0]), int(c[1]),int(c[2]), int(c[3])
                        sub_img = blankimg[ymin+100:ymax+100, xmin+100:xmax+100]
                        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
                        res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
                    blankimg[ymin+100:ymax+100, xmin+100:xmax+100] = res
            spacer = 30
            for i,v in tag_codes.items():
                cv2.putText(blankimg,f"{str(i)} = {str(v[0])}",(width+110,130+spacer),0,5e-3 * 180,color_dic[v[1]],3)
                spacer+=35
            spacer+=10
            cv2.putText(blankimg,'Mouse Head Other Mouse bpt',(width+110,130+spacer),0,5e-3 * 150,(0,0,0),3)
            if dlc_bpts:
                bpt_plot=bpts[frame_count]
                for bpt in bpt_plot: 
                   xc, yc = int(bpt[0]),int(bpt[1])
                   cv2.circle(blankimg, (xc+100, yc+100), 5, (0, 255, 0), -1) 
                if len(tags)>0:
                    dlc_dist=[i[frame_count] for i in dlc_columns]
                    for dic,color in zip(dlc_dist,dlc_colors):
                        for idx,value in dic.items():
                            cv2.line(blankimg, (int(value[0])+100,int(value[1])+100), (int(value[2])+100,int(value[3])+100), \
                                     (color[0],color[1],color[2]), thickness=2, lineType=cv2.LINE_8)
                    spacer+=50
                    for name, color in zip(itertools.combinations(tags, 2),dlc_colors):
                        col_name=f'{name[0]}_{name[1]}:'
                        cv2.putText(blankimg,col_name,(width+110,130+spacer),0,5e-3 * 180,(0,0,0),1)
                        cv2.line(blankimg,(int(width+500),int(120+spacer)),(int(width+600),int(120+spacer)),(color[0],color[1],color[2]), thickness=2, lineType=4)
                        spacer+=50
            out.write(blankimg)
            frame_count+=1
            pbar.update(1)
        else:
            break
            df_tracks_out['dbpt_pairs']=distances_doc_f
    out.release()
    vid.release()
    
def create_validation_Video(folder,df1,tags,config_dic,output=None,plot_readers=True):
    if output is None:
        output = folder
    #output='/media/tony/data/data/test_tracks/vertification/older_coords/vertifications'
    RFID_coords=config_dic['RFID_readers']
    entrance_reader=config_dic['entrance_reader']
    if entrance_reader != None:
        entrance_reader=entrance_reader
    else:
        entrance_reader=None
    RFID_coords=config_dic['RFID_readers']
    dics={'sort_tracks':eval,'RFID_tracks':eval,'Correction':eval,'RFID_matched':eval,'Matching_details':eval}#'bpts':eval}
    df1=pd.read_csv(folder+'/RFID_tracks.csv',converters=dics)
    max_mice= len(tags)
    colors=['green','blue','purple','black','orange','red','yellow','aqura','magenta','lbrown']
    tag_codes={tag: num+1 for num, tag in zip(range(len(tags)),tags)}
    tag_codes={i:[tag_codes[i],z] for i,z in zip(tag_codes.keys(),colors)}
    color_dic={'green':(0,255,0),'blue':(255,0,0),'purple':(128,0,128),'black':(0,0,0),
               'orange':(155,140,0),'red':(0,0,255),'yellow':(0,255,255),'aqura':(212,255,127),
               'magenta':(255,0,255),'lbrown':(181,228,255)}
    yolo_bboxes=df1.sort_tracks.values
    rfid_tracks=df1.RFID_tracks.values
    #bpts=df1['bpts'].to_list()
    RFID_readings={i:eval(v) for i,v in enumerate(df1.RFID_readings.values) if type(v) is str}
    validation_frames=[frame for frame in RFID_readings.keys()]
    corrections={i:v for i,v in enumerate(df1.Correction) if v!= []}
    correction_frames=[frame for frame in corrections.keys()]
    matched={i:v for i,v in enumerate(df1.RFID_matched.values) if v!= []}
    matched_frames=[frame for frame in matched.keys()]
    match_details={i:v for i,v in enumerate(df1.Matching_details.values) if v!= []}
    md_frames=[frame for frame in match_details.keys()]
    #corrections=df1.Correction.values
    #matched=df1.RFID_matched.values
    vid=cv2.VideoCapture(folder+'/raw.mp4')
    vid_length=int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output=output+'/'+os.path.basename(folder)+'_v_r.avi'
    out= cv2.VideoWriter(output, codec, fps, (1920, 1080),True)#1920,1260
    frame_count=0
    pbar=tqdm(total=vid_length,position=0,leave=False,desc='Writing Validation Video')
    while vid.isOpened():
        ret,img=vid.read()
        if ret and frame_count<vid_length-1:   
            if len([i for i in validation_frames if i>=frame_count])>0:
                RFID_frame=[i for i in validation_frames if i>=frame_count][0]
            else:
                RFID_frame=0
            if len([i for i in correction_frames if i>=frame_count])>0:
                correction_frame=[i for i in correction_frames if i>=frame_count][0]
            else:
                correction_frame=0
            if len([i for i in matched_frames if i>=frame_count])>0:
                matched_frame=[i for i in matched_frames if i>=frame_count][0]
            else:
                 matched_frame=0
            if len([i for i in md_frames if i>=frame_count])>0:
                md_frame=[i for i in md_frames if i>=frame_count][0]
            else:
                md_frame=0
            rfid_tracker=rfid_tracks[frame_count]
            yolo_dets=yolo_bboxes[frame_count]
            #corrections_display=corrections[frame_count]
            #matched_display=matched[frame_count]
            #bpt_plot=bpts[frame_count]
            img_yolo=img.copy()
            img_rfid=img.copy()
            for objects in rfid_tracker:
                xmin, ymin, xmax, ymax, index = int(objects[0]), int(objects[1]),\
                    int(objects[2]), int(objects[3]), int(objects[4])
                m_display=tag_codes[index][0]
                m_color=color_dic[tag_codes[index][1]]
                cv2.rectangle(img_rfid, (xmin, ymin), (xmax, ymax), m_color, 3)   
                cv2.putText(img_rfid, str(m_display), (xmin, ymin-20), 0, 5e-3 * 200, m_color, 3)
            #for bpt in bpt_plot: 
            #   xc, yc = int(bpt[0]),int(bpt[1])
            #   cv2.circle(img_rfid, (xc, yc), 5, (155,140,0), -1) 
            for objects in yolo_dets:
                #print(objects)
                xmin, ymin, xmax, ymax, index = int(objects[0]), int(objects[1]),\
                    int(objects[2]), int(objects[3]), int(objects[4])
                cv2.rectangle(img_yolo, (xmin, ymin), (xmax, ymax), (0,255,0), 3)    
                cv2.putText(img_yolo, str(index), (xmin, ymin+10), 0, 5e-3 * 200, (0,0,255), 3)
            if plot_readers:
                for i,v in RFID_coords.items():
                    if i!=entrance_reader:
                        xmin, ymin, xmax, ymax=v[0],v[1],v[2],v[3]
                        cv2.rectangle(img_rfid, (xmin, ymin), (xmax, ymax), (0,0,255), 3)
                        cent_point=bbox_to_centroid(v)
                        cv2.putText(img_rfid,f"{str(i)}",(int(cent_point[0]),int(cent_point[1])),0, 5e-3 * 200,(0,0,255),3)
            if 3*width +100<2800:
                width_b=2300
            else:
                width_b=3*width+300
            blankimg=255*np.ones(shape=[height+400,width_b,3],dtype=np.uint8)
            blankimg[200:height+200,100:width+100]=img_yolo
            blankimg[200:height+200,width+150:2*width+150]=img_rfid
            cv2.putText(blankimg,f"Frame: {str(frame_count)}",(int(0.5*width),50),0, 5e-3 * 250,(0,0,255),2)
            cv2.putText(blankimg,f"Maximum of Mice in Video: {max_mice}",(int(0.5*width)+275,50),0, 5e-3 * 250,(0,0,255),2)
            cv2.putText(blankimg,f"Current Mice in Video: {len(yolo_dets)}",(int(0.5*width)+950,50),0, 5e-3 * 250,(0,0,255),2)
            cv2.putText(blankimg,'SORT ID Tracking',(100,150),0, 5e-3 * 250,(0,0,255),5)
            cv2.putText(blankimg,'RFID Tracking',(150+int(width),150),0, 5e-3 * 250,(0,0,255),5)
            if RFID_frame != [] and RFID_frame !=0:
                spacer=0
                for i in RFID_readings[RFID_frame]:
                    if i[1] in tags:
                        if i[0] == entrance_reader:
                            reader_display='Entrancer'
                        else:
                            reader_display=i[0]
                        tag_display=tag_codes[i[1]][0]
                        cv2.putText(blankimg,f'Frame {RFID_frame}: reader {str(reader_display)}    tag read {tag_display}', \
                                    (int(1.4*width),int(height+300+spacer)),0,5e-3 * 210,(0,0,255),3)#1.4*height+spacer
                        spacer+=50        
            cv2.putText(blankimg,'RFID Tag Codes',(2*width+175,150),0,5e-3 * 210,(0,0,255),5)
            spacer=0
            for i,v in tag_codes.items():
                cv2.putText(blankimg,f"{str(i)} = {str(v[0])}",(2*width+175,230+spacer),0,5e-3 * 180,color_dic[v[1]],3)
                spacer+=35
            spacer+=10
            cv2.putText(blankimg,'RFID-SID Matching Log',(2*width+175,250+spacer),0,5e-3 * 210,(0,0,0),4)
            spacer+=35
            if correction_frame != []  and correction_frame !=0:
                #spacer=0
                for i in corrections[correction_frame]:
                    for item,value in i[3].items():
                        if value != None:
                            cv2.putText(blankimg,f'Correction on SID {item} from frame {value} to frame {i[2]} ', \
                                        (2*width+175,250+spacer),0,5e-3 * 200,(0,0,255),2)#1.4*height+spacer
                        spacer+=50
            if matched_frame != [] and matched_frame != 0:
                #spacer=0
                for i in matched[matched_frame]:
                    if type(i[1])!= str:
                        if type(i[2])!=str:
                            tag_display=tag_codes[i[1]][0]
                            #print(i)
                            cv2.putText(blankimg,f'Frame: {matched_frame} {tag_display} matched to sid: {i[0]} ', \
                                        (2*width+175,250+spacer),0,5e-3 * 200,(0,0,255),2)#1.4*height+spacer
                            spacer +=50
                        else:
                            tag_display=tag_codes[i[1]][0]
                            cv2.putText(blankimg,f'Frame {matched_frame}:Last tag {tag_display} matched to sid: {i[0]} ', \
                                        (2*width+175,250+spacer),0,5e-3 * 180,(0,0,255),2)#1.4*height+spacer
                            spacer +=50
                    else:
                        tag_display=tag_codes[i[2]][0]
                        cv2.putText(blankimg,f'Frame {matched_frame}: {tag_display} {i[1]} matched to sid: {i[0]} ', \
                                    (2*width+175,250+spacer),0,5e-3 * 180,(0,0,255),2)#1.4*height+spacer
                        spacer+=50        
            if  md_frame != [] and md_frame !=0:
                #print(match_details[md_frame])
                for i in match_details[md_frame]:
                        #print(i)
                        tag_display=tag_codes[int(i[1])][0]
                        #print(i)
                        cv2.putText(blankimg,f'frame {md_frame}: {i[0]} {tag_display}', \
                                    (2*width+175,250+spacer),0,5e-3 * 200,(0,0,255),2)#1.4*height+spacer
                        spacer+=30        
            """
            if corrections_display !=[]:
                print(corrections_display)
                for display in corrections_display:
                    cv2.putText(blankimg,f"{display[0],display[1]}",(2*width+175,130+spacer),0,5e-3 * 180,(255,255,255),3)
                    spacer+=100
            if matched_display != []:
                print(matched_display)
                for display in matched_display:
                    print(display)
                    cv2.putText(blankimg,f"{display[0],display[1],display[2]}",(2*width+175,130+spacer),0,5e-3 * 180,(255,255,255),3)
            """
            blankimg=cv2.resize(blankimg,(1920, 1080))
            out.write(blankimg)
            frame_count+=1
            pbar.update(1)
        else:
            break
    out.release()
    vid.release()
