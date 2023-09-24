

import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import itertools
from prt_utils.track_utils import *
from scipy.optimize import linear_sum_assignment
from prt_utils.sort import Sort
from itertools import chain
from decord import VideoReader
from decord import cpu, gpu
import os
pd.options.mode.chained_assignment = None


"""
Pandas based processing functionsimport multiprocessing 
"""

def RFID_readout(pathin,config_dict_analysis,n_mice):
    """
    Loads to RFID csv file in to pandas dataframe[mm.rfid2bpts
    df_RFID_cage=rm.RFID_readout(FLAGS.data,0)# change for entering specific reader as entrance reader
        1.pathin: path of the rfid files ( data_all.csv and text.csv)
        2. ent_entrance reader number
    """
    #add datetime backin 
    if n_mice ==1:
        try:
            df2=pd.read_csv(f'{pathin}/RFID_data_all.csv',index_col=False)
            df2.Time=pd.to_datetime(df2['Time'],format="%Y-%m-%d_%H:%M:%S.%f")
            df2['Time']=df2['Time'].astype(np.int64)/10**9
        except Exception:
            df2=pd.read_csv(f'{pathin}/timestamps.csv',index_col=False,skiprows=1)
            df2['timestamp']=(df2['timestamp']-df2.iloc[0]['timestamp'])/1000000
            with open(f'{pathin}/timestamps.csv') as f:
                first_line = f.readline()
            starttime=float(first_line[10:-2])
            df2['Time']=df2['timestamp'] + starttime
    else:
        print(config_dict_analysis)
        ent_reader=config_dict_analysis['entrance_reader']
        time_thresh=config_dict_analysis['entrance_time_thres']
        try:
            df1=pd.read_csv(f'{pathin}/text.csv')
            df1.Timestamp=pd.to_datetime(df1['Timestamp'],format="%Y-%m-%d %H:%M:%S.%f")
            df1['Timestamp']=df1['Timestamp'].astype(np.int64)/10**9
        except Exception:
            df1=pd.read_csv(f'{pathin}/RFID_reads.csv',index_col=False)
            df1=df1.rename(columns={'timestamp':'Timestamp'})
        try:
            df2=pd.read_csv(f'{pathin}/RFID_data_all.csv',index_col=False)
            df2.Time=pd.to_datetime(df2['Time'],format="%Y-%m-%d_%H:%M:%S.%f")
            df2['Time']=df2['Time'].astype(np.int64)/10**9
        except Exception:
            df2=pd.read_csv(f'{pathin}/timestamps.csv',index_col=False,skiprows=1)
            df2['Timestamp']=(df2['Timestamp']-df2.iloc[0]['Timestamp'])/1000000
            with open(f'{pathin}/timestamps.csv') as f:
                first_line = f.readline()
            starttime=float(first_line[10:-2])
            df2['Time']=df2['Timestamp'] + starttime
        #print(df1.columns)
        if ent_reader is not None:
            df_ent_reader=df1.query(f'Reader =={str(ent_reader)}')
            ent_times=df_ent_reader.diff()#.Timestamp.apply(lambda x: x.total_seconds())
            df1=df1.drop(df1.index[ent_times[ent_times.Timestamp<time_thresh].index],inplace=False)
        df1['Readings']=df1.drop(columns=['Timestamp']).values.tolist()
        df1=df1.reset_index()
        df2=df2.reset_index()
        idx=np.searchsorted(df2['Time'], df1['Timestamp'],side='right').tolist()
        df2['RFID_readings']=np.nan
        df2['RFID_readings']=df2['RFID_readings'].astype('object')
        for i in range(len(idx)): 
            if idx[i] ==len(df2):
                idx[i]=len(df2)-1
            else:
                pass
        for i in range(len(idx)):
            if np.isnan(df2.iloc[idx[i]]['RFID_readings']).all():
                df2.at[idx[i],'RFID_readings']=[df1.iloc[i]['Readings']]
            else:
                original=df2.iloc[idx[i]]['RFID_readings']
                original.append(df1.iloc[i]['Readings'])
                df2.at[idx[i],'RFID_readings']=original
        df2=df2.rename(columns={'Frame':'frame'})
        df2=df2.reset_index()
        df2=df2.drop(columns=['index'])
        try:
            df2=df2.drop(columns=['RFID_0', 'RFID_1', 'RFID_2', 'RFID_3', 'RFID_4', 'RFID_5'])
        except Exception:
            pass
        df2['frame']=range(len(df2))
    #print(df2.columns)
    return df2

def sort_tracks_generate(bboxes,resolution,area_thres,max_age,min_hits,iou_threshold,n_mice):
    #memory issues?
    #bboxes=[track[:5] for track_list in bboxes for track in track_list]
    id_tracks=[]
    unmatched_predicts=[]
    mot_tracker=Sort(n_mice,max_age,min_hits,iou_threshold)
    mot_tracker.reset_count()   
    pbar=tqdm(total=len(bboxes),position=0,leave=True,desc='Assigning Sort ID to bboxes')
    count=0
    for i in bboxes:
        ds_boxes=np.asarray(i)
        sort_output= mot_tracker.update(ds_boxes)
        if len(sort_output) != 0:
            trackers,unmatched_predict=sort_output[0],sort_output[1]
            id_tracks.append(trackers)#.tolist())
            unmatched_predicts.append(np.asarray(unmatched_predict))
        else:
            unmatched_predicts.append(np.asarray([]))
            id_tracks.append(np.asarray([]))
        count+=1
        pbar.update(1)
    return id_tracks,unmatched_predicts

def mouse_limiter(bb,n_mice):
    if len(bb)> n_mice:
        bb.sort(key=lambda x:x[4], reverse=True)
        bb=bb[:n_mice]
    return bb
    
def read_yolotracks(pathin,config_dict_analysis,config_dict_tracking,df_RFID,n_mice):
    RFID_coords=config_dict_analysis['RFID_readers']
    entrance_reader=config_dict_analysis['entrance_reader']
    ent_thres=config_dict_analysis['entrance_distance']
    interaction_thres=config_dict_tracking['interaction_thres']
    ent_time=config_dict_analysis['entrance_time_thres']
    max_age=config_dict_tracking['max_age']
    min_hits=config_dict_tracking['min_hits']
    iou_threshold=config_dict_tracking['iou_threshold']
    resolution=config_dict_tracking['resolution']
    columns=['frame','bboxes','motion_roi']
    dics={i: eval for i in columns}
    df_tracks=pd.read_csv(pathin+'/'+'yolo_dets.csv',converters=dics)
    #####################################################
    ####################################################
    bboxes=[track for track_list in df_tracks['bboxes'].values for track in track_list]
    #print(bboxes)
    areas=[bbox_area(bbox) for bbox in bboxes]
    area_thresh=[0.75*min(areas),max(areas)]
    df_tracks['bboxes']=[mouse_limiter(bbs,n_mice) for bbs in df_tracks['bboxes'].values]
    if np.all(df_tracks.frame==0):
        df_tracks['frame']=[i+1 for i in range(len(df_tracks))]
    sort_tracks,unmatched_predicts=sort_tracks_generate(df_tracks['bboxes'].values,resolution,area_thresh,
                                                        max_age,min_hits,iou_threshold,n_mice)
    df_tracks['unmatched_predicts']=unmatched_predicts
    #df_tracks['unmatched_predicts']=[float_int(x) for x in df_tracks['unmatched_predicts'].values]
    df_tracks['sort_tracks']=sort_tracks
    #df_tracks['sort_tracks']=[float_int(x) for x in df_tracks['sort_tracks'].values]
    df_tracks=bbox_revised(df_tracks,df_RFID,RFID_coords,entrance_reader,
                           ent_time,interaction_thres,ent_thres,n_mice,pathin,area_thresh,resolution)
    #df_tracks['sort_tracks']=df_tracks['sort_tracks'].apply(array2list)
    df_tracks['sort_tracks']=[float_int(x) for x in df_tracks['sort_tracks'].values]
    if entrance_reader is None:
        pass
    else:
        df_tracks['Entrance']=[track_splitter(x,RFID_coords,entrance_reader,ent_thres)[0] for x in df_tracks['sort_tracks'].values]
        df_tracks['Cage']=[track_splitter(x,RFID_coords,entrance_reader,ent_thres)[1] for x in df_tracks['sort_tracks'].values]
        df_tracks['sort_entrance_dets']=[len(x) for x in df_tracks['Entrance'].values]
        df_tracks['Entrance_ids']=[get_ids(x) for x in df_tracks['Entrance'].values]
        df_tracks['Cage_ids']=[get_ids(x) for x in df_tracks['Cage'].values]
        df_tracks['sort_cage_dets']=[len(x) for x in df_tracks['Cage'].values]
    df_tracks['track_id']=[get_ids(x) for x in df_tracks['sort_tracks'].values]
    df_tracks['ious']=[iou_tracks(x) for x in df_tracks['sort_tracks'].values]
    #df_tracks.to_csv('/media/tony/data/data/test_tracks/vertification/older_coords/vertifications/temp.csv')
    return df_tracks

def reconnect_tracks_ofa(df_tracks,n_tags):# add in configdic settings
    """
    maymove this to detection processing
    need to optimize for better performance
    # work in leap distance
    
    """
    id_range=[ids+1 for ids in range(n_tags)]
    pbar=tqdm(total=len(df_tracks),position=0, leave=True,desc='Reconnecting IDs:')
    for i in range(len(df_tracks)):
        #get the df values once insteasd of mutiple calls
        tracked_id = df_tracks.iloc[i]['track_id']
        if len(tracked_id) >0:
            new_ids=[new_id for new_id in tracked_id if new_id not in id_range]
            if len(new_ids)>0:
                missing_ids=list(set(id_range)-set(df_tracks.iloc[i]['track_id']))
                idx=[]
                #get rid of append
                for ids in missing_ids:
                    idx=[]
                    for ind,sids in enumerate(reversed(df_tracks.iloc[:i]['track_id'].values)):
                        if ids in sids:
                            idx.append(ind)
                            break
                    #index_recent=max([ind for ind,sids in enumerate(df_tracks.iloc[:i]['track_id']) if ids in sids])
                    #idx.append(index_recent)
                old_centroids=[]
                for ids,ind in zip(missing_ids,idx):
                    old_centroids+=[bbox_to_centroid(track) for track in df_tracks.iloc[ind]['sort_tracks'] if track[4]==ids]
                c=[track for track in df_tracks.iloc[i]['sort_tracks'] if track[4] in new_ids]
                c=[bbox_to_centroid(track) for track in c]
                distance_matrix=[]
                for centroid in c:
                    distance_matrix.append([Distance(centroid,z) for z in old_centroids])
                rnd,col = linear_sum_assignment(distance_matrix)
                for ids,ind in zip(new_ids,col):
                    if distance_matrix[new_ids.index(ids)][ind] <200:
                        tracks=[track for track in df_tracks.iloc[i]['sort_tracks'] if track[4]!=ids]
                        track_re=[track for track in df_tracks.iloc[i]['sort_tracks'] if track[4]==ids][0]
                        track_add=[[track_re[0],track_re[1],track_re[2],track_re[3],missing_ids[ind]]]
                        tracks=tracks+track_add
                        df_tracks.at[i,'sort_tracks']=tracks
                        df_tracks.at[i,'track_id']=[track[4] for track in tracks]
                    else:
                        #git rid of append
                        id_range.remove(missing_ids[ind])
                        id_range.append(ids)
        pbar.update(1)
    df_tracks['track_id']=[get_ids(x) for x in df_tracks['sort_tracks'].values]
    df_tracks['ious']=[iou_tracks(x) for x in df_tracks['sort_tracks'].values]
    return df_tracks


def get_RFID_tracked(x):
    tracks_marked=[i[:4] for i in x]
    appearance_dict={}
    if len(tracks_marked)>1:
        for i,v in enumerate(tracks_marked):
            if tuple(v) in appearance_dict:
                appearance_dict[tuple(v)].append(i)
            else:
                appearance_dict[tuple(v)]=[i]
        mutiple_marked={i:v for i,v in appearance_dict.items() if len(v)>1}
    else:
        return {}
    return mutiple_marked



def get_reconnects(df_tracks):
    cage_id_list=get_unique_ids(df_tracks['Cage_ids'].to_list())
    Entrance_id_list=get_unique_ids(df_tracks['Entrance_ids'].to_list())
    cage_ids_to_check=[i for i in cage_id_list if i in Entrance_id_list]
    cage_ids_to_recconnect=np.array([i for i in cage_id_list if i not in Entrance_id_list])
    cage_appearances= df_tracks['Cage_ids'].values
    entrance_appearances=df_tracks['Entrance_ids'].values
    id_reconnect={}
    for i in cage_ids_to_check:
        for ind,ids in enumerate(cage_appearances):
            if i in ids:
                cage_first_appear=ind
                break
        for ind,ids in enumerate(entrance_appearances):
            if i in ids:
                 entrance_first_appear=ind
                 break
        #cage_first_appear=[ind for ind,ids in enumerate(cage_appearances) if i in ids][0]
        #entrance_first_appear=[ind for ind,ids in enumerate(entrance_appearances) if i in ids][0]
        #entrance_first_appear
        if  entrance_first_appear>cage_first_appear and cage_first_appear>100:
            id_reconnect[i]=cage_first_appear
        else:
            pass
    for i in cage_ids_to_recconnect:
        #cage_first_appear=[ind for ind,ids in enumerate(cage_appearances) if i in ids][0]
        for ind,ids in enumerate(cage_appearances):
            if i in ids:
                cage_first_appear=ind
        if cage_first_appear>100:
            id_reconnect[i]=cage_first_appear
    return id_reconnect

def track_exist_af(sort_id,frame,df):
    checklisk_entrance=df.iloc[frame:].apply(lambda x:check_id_inlist(x['sort_ids'],sort_id),axis=1).to_list()
    if True in checklisk_entrance:
        return False
    else:
        return True


def replace_track_leap(frame,sort_id_old,sort_id_new,df_tracks):
    ind_list=[ind for ind, sids in enumerate(df_tracks.iloc[frame:]['track_id'].values) if sort_id_new in sids]
    msg=f'\nStarting at frame {frame} to reassciate SORT ID {sort_id_new} to {sort_id_old} '
    pbar=tqdm(total=len(ind_list),position=0, leave=True,desc=msg)
    for i in ind_list:
        tracks_remain=[z for z in df_tracks.iloc[frame+i]['sort_tracks'] if z[4] !=sort_id_new]
        track_new=[z for z in df_tracks.iloc[frame+i]['sort_tracks'] if z[4] ==sort_id_new][0]
        track_new[4]=sort_id_old
        tracks_remain.append(track_new)
        df_tracks.at[frame+i,'sort_tracks']=tracks_remain
        pbar.update(1)
    return df_tracks

def reconnect_leap(reconnect_ids,df_tracks,max_distance):
    track_dict={}
    for i,v in reconnect_ids.items():
        frame_interest_dets=df_tracks.iloc[v]['sort_cage_dets'] 
        index_n= list(df_tracks.iloc[v-10:v].query(f'sort_cage_dets=={frame_interest_dets}').index)
        if len(index_n)<1:
            pass
        else:
            index_i=index_n[-1]
            track_exist_after=[z for z in df_tracks.iloc[index_i:]['Cage_ids'] if i in z]
            if len(track_exist_after)>3:
                lost_track=[z for z in df_tracks.iloc[index_i]['Cage_ids']\
                            if z not in get_unique_ids(df_tracks.iloc[v:]['Cage_ids'].tolist())] # the lost track dissappears after
                if len(lost_track)>0:
                    sort_track_i=[z for z in df_tracks.iloc[v]['sort_tracks'] if z[4]==i][0]
                    new_centroid=bbox_to_centroid(sort_track_i)
                    old_centroids=[bbox_to_centroid(z) for z in df_tracks.iloc[index_i]['sort_tracks'] if z[4] in lost_track]
                    distances=[Distance(new_centroid,z) for z in old_centroids]
                    ious=[iou(z,sort_track_i) for z in df_tracks.iloc[index_i]['sort_tracks'] if z[4] in lost_track]
                    #check if the index had an iou with another 
                    iou_dic={sids:iou  for sids,iou in df_tracks.iloc[index_i]['ious'].items() if lost_track[distances.index(min(distances))]\
                             in sids and iou >0.1}
                    if min(distances) <max_distance and len(iou_dic)==0:
                        track_dict[i]=[v,lost_track[distances.index(min(distances))]]
                    else:
                        pass
    return track_dict

def replace_track_leap_df(track_dict,df_tracks,config_dict_analysis):
    RFID_coords=config_dict_analysis['RFID_readers']
    entrance_reader=config_dict_analysis['entrance_reader']
    ent_thres=config_dict_analysis['entrance_distance']
    for i,v in track_dict.items():
        df_tracks=replace_track_leap(v[0],v[1],i,df_tracks)
    df_tracks['track_id']=[get_ids(x) for x in df_tracks['sort_tracks'].values]
    df_tracks['ious']=[iou_tracks(x) for x in df_tracks['sort_tracks'].values]
    df_tracks['Entrance']=[track_splitter(x,RFID_coords,entrance_reader,ent_thres)[0] for x in df_tracks['sort_tracks'].values]
    df_tracks['sort_entrance_dets']=[len(x) for x in df_tracks['Entrance'].values]
    df_tracks['Cage']=[track_splitter(x,RFID_coords,entrance_reader,ent_thres)[1] for x in df_tracks['sort_tracks'].values]
    df_tracks['Cage_ids']=[get_ids(x) for x in df_tracks['Cage'].values]
    df_tracks['Entrance_ids']=[get_ids(x) for x in df_tracks['Entrance'].values]
    df_tracks['sort_cage_dets']=[len(x) for x in df_tracks['Cage'].values]
    return df_tracks



def get_interaction(untracked,lost_tracks,ious,tags):
    interaction_dic={tag:[] for tag in tags}
    for tag in tags:
        iou_dic={ind  for ind,values in ious.items() if tag in ind and values>0}
        if len(iou_dic) >0:
            interaction_tag=[id for ind in iou_dic for id in ind if id!=tag]
            interaction_tag=[id if id in tags else 'UK' for id in interaction_tag]
        else:
            interaction_tag=[]
        interaction_dic[tag]=interaction_tag
    if len(untracked)==len(lost_tracks):
        id_lost=[track[4] for track in lost_tracks]
        if len(untracked) ==2:
            if [value for ind,value in ious.items() if id_lost[0] in ind and id_lost[1] in ind][0]>0:
                interaction_dic[untracked[0]]=[untracked[1]]
                interaction_dic[untracked[1]]=[untracked[0]]
    return interaction_dic



def interaction2dic(df_tracks_out,tags,slack):
    df_tracks_out['Tracked']=[get_ids(x) for x in df_tracks_out.RFID_tracks.values]                                
    df_tracks_out['UnTracked']=[list(set(tags)-set(x)) for x in df_tracks_out['Tracked'].values]
    df_tracks_out['Interaction_tracks']=df_tracks_out['RFID_tracks']+df_tracks_out['lost_tracks']
    df_tracks_out['Interaction_tracks']=[apply_slack(i,slack) for i in df_tracks_out['Interaction_tracks'].values]
    df_tracks_out['ious_interaction']=[iou_tracks(x) for x in df_tracks_out['Interaction_tracks'].values]
    df_tracks_out['Interactions']=[get_interaction(i,z,y,tags) for i,z,y in zip (df_tracks_out['UnTracked'].values,df_tracks_out['lost_tracks'].values,\
                                                                        df_tracks_out['ious_interaction'].values)]
    return df_tracks_out


def spontanuous_bb_remover(reconnect_ids,df_tracks,config_dict_analysis,config_dict_tracking):
    # removes fast appearing bbs , usually false negatives
    iou_min_bb=config_dict_tracking['iou_min_sbb_checker']
    RFID_coords=config_dict_analysis['RFID_readers']
    fpbb_thres=config_dict_tracking['sbb_frame_thres']
    entrance_reader=config_dict_analysis['entrance_reader']
    ent_thres=config_dict_analysis['entrance_distance']
    id2remove=[]
    for sort_id, frame in reconnect_ids.items():
        iou_dic={i:v  for i,v in df_tracks.iloc[frame]['ious'].items() if sort_id in i}
        if len(iou_dic)>0:
            iou_max=iou_dic[max(iou_dic,key=iou_dic.get)]
            if iou_max<iou_min_bb:
                pass
            else:
                index_checklist=[ind for ind,sids in enumerate(df_tracks.iloc[frame:]['track_id'].values) if sort_id in sids]
                if len(index_checklist) >fpbb_thres:
                    for ind in index_checklist:# check if the first 7 frames if there are 
                        iou_dic2={i:v  for i,v in df_tracks.iloc[frame+ind]['ious'].items() if sort_id in i}
                        if len(iou_dic2)<2:
                            pass
                        else:
                            key_check=max(iou_dic2,key=lambda k:iou_dic2[k])
                            id_canadidate=[i for i in key_check if i !=sort_id][0]
                            track_2check=[i for i in df_tracks.iloc[frame+ind]['sort_tracks'] if i[4]==id_canadidate]
                            if len(track_2check)==1:
                                track_2check=[i for i in df_tracks.iloc[frame+ind]['sort_tracks'] if i[4]==id_canadidate][0]
                                track_2check2=[i for i in df_tracks.iloc[frame+ind]['sort_tracks'] if i[4]==sort_id][0]
                                if track_2check[0]<track_2check2[0] and  track_2check[1]<track_2check2[1] and track_2check[2]>track_2check2[2] \
                                    and track_2check[3]>track_2check2[3]:
                                        id2remove.append(sort_id)
                                        tracks_remain=[z for z in df_tracks.iloc[frame+ind]['sort_tracks'] if z[4] !=sort_id]
                                        df_tracks.at[frame+ind,'sort_tracks']=tracks_remain
                                        print(f'Removed sort id {sort_id} at frame {frame+ind} for complete overlap')
                else:
                    for ind in index_checklist:
                        tracks_remain=[z for z in df_tracks.iloc[frame+ind]['sort_tracks'] if z[4] !=sort_id]
                        df_tracks.at[frame+ind,'sort_tracks']=tracks_remain
                        id2remove.append(sort_id)
    print(f'Removed sort ids {id2remove} for possible false positives')
    id2remove=list(set(id2remove))
    df_tracks['track_id']=[get_ids(x) for x in df_tracks['sort_tracks'].values]
    df_tracks['ious']=[iou_tracks(x) for x in df_tracks['sort_tracks'].values]
    if entrance_reader is None:
        pass
    else:
        df_tracks['Entrance']=[track_splitter(x,RFID_coords,entrance_reader,ent_thres)[0] for x in df_tracks['sort_tracks'].values]
        df_tracks['sort_entrance_dets']=[len(x) for x in df_tracks['Entrance'].values]
        df_tracks['Cage']=[track_splitter(x,RFID_coords,entrance_reader,ent_thres)[1] for x in df_tracks['sort_tracks'].values]
        df_tracks['Cage_ids']=[get_ids(x) for x in df_tracks['Cage'].values]
        df_tracks['Entrance_ids']=[get_ids(x) for x in df_tracks['Entrance'].values]
        df_tracks['sort_cage_dets']=[len(x) for x in df_tracks['Cage'].values]
    return df_tracks,id2remove








def coverage(df):#,ent_thres,entrance_reader,RFID_coords):
    a=sum([len(i) for i in df.RFID_tracks])
    b=sum([len(i) for i in df.sort_tracks])
    cov=a/b*100
    print(f'{cov} % of mice detected were matched with an RFID tag')
    return cov


'''
Removes duplicate loggins in the dataframe
'''
def duplicate_remove(list_dup):
    l1=[]
    [l1.append(i) for i in list_dup if i not in l1]
    return l1

'''
splits the sort_tracks into ones in cage vs near entrance based on ent_thres
'''
def track_splitter(sort_tracks,RFID_coords,entrance_reader,ent_thres):
    sort_entrance_distance=[distance_to_entrance(i,RFID_coords,entrance_reader) for i in sort_tracks]
    sort_entrance_tracks=[i for i,v in enumerate(sort_entrance_distance) if v <=ent_thres]
    sort_entrance_tracks=[v for i,v in enumerate(sort_tracks) if i in sort_entrance_tracks] 
    sort_cage_tracks=[i for i,v in enumerate(sort_entrance_distance) if v>ent_thres]
    sort_cage_tracks=[v for i, v in enumerate(sort_tracks) if i in sort_cage_tracks]
    return sort_entrance_tracks, sort_cage_tracks
'''
Reads the tracks processed into a dataframe
'''
def df_tracks_read(path):
    columns=['frame_count','Sort_dets','Sort_track_ids','Sort_tracks','sort_cage_dets',
                                 'sort_cage_ids','sort_cage_tracks','sort_entrance_dets','sort_entrance_ids',
                                 'sort_entrance_tracks','iou','track_id','tracks','RFID_readings']
    dics={i: eval for i in columns}
    df = pd.read_csv(path, converters=dics)
    df=df.set_index('Unnamed: 0')
    return df


def iou_tracks(sort_tracks):
    iou_index=[]
    iou_area=[]
    for combinations in itertools.combinations(sort_tracks,2):
        iou_index.append((combinations[0][4],combinations[1][4]))
        iou_area.append(iou(combinations[0],combinations[1]))
    iou_dictionary= {i:v for i,v in zip(iou_index,iou_area)}
    return iou_dictionary


def Combine_RFIDreads(df,df_RFID_cage):
    length=min([len(df),len(df_RFID_cage)])
    df=df[:length]
    df_RFID_cage=df_RFID_cage[:length]
    #print(df_RFID_cage.columns)
    #print(df.columns)
    df=pd.merge(df,df_RFID_cage, on='frame')
    df['RFID_tracks']=[list() for i in range(len(df.index))]
    df['RFID_matched']=[list() for i in range(len(df.index))]
    return df

#write function


def RFID_matching(df_tracks,tags,config_dict_analysis,folder_path):
    entrance_reader=config_dict_analysis['entrance_reader']
    RFID_coords=config_dict_analysis['RFID_readers']
    entr_frames=config_dict_analysis['entr_frames']
    correct_iou=config_dict_analysis['correct_iou']
    reader_thresh=config_dict_analysis['reader_thres']
    RFID_dist=config_dict_analysis['RFID_dist']
    ent_thres=config_dict_analysis['entrance_distance']
    df_tracks['sort_tracks']=[sorted(tracks,key= lambda x: x[4]) for tracks in df_tracks['sort_tracks']]
    validation_frames=[f for f in df_tracks[df_tracks.RFID_readings.notnull()].index]
    validation_frames.sort()
    #pd.DataFrame(validation_frames).to_csv('/media/tony/data/data/test_tracks/vertification/older_coords/vertifications/v1.csv')
    df_tracks['Correction']=[[] for i in range(len(df_tracks))]
    df_tracks['Matching_details']=[[] for i in range(len(df_tracks))]
    path=folder_path+'/matching_process_cage.csv'
    with open(path,'w') as file:
        file.write('frame,reader,readings,reader_coord,sort_tracks,Dist_r,iou_r,track_exist\n')
    if entrance_reader is None:
        pass
    else:
        path=folder_path+'/matching_process_entrance.csv'
        with open(path,'w') as file:
            file.write('frame,readings,past_ids,future_ids\n')
        entrance_val=list(set([i for i in validation_frames for x in df_tracks.iloc[i]['RFID_readings'] if x[0]==entrance_reader]))
        entrance_val.sort()
    cage_val=list(set([i for i in validation_frames for x in df_tracks.iloc[i]['RFID_readings'] if x[0]!=entrance_reader]))
    cage_val.sort()
    if entrance_reader is None:
        pass
    else:
        msg='Starting to match RFID readings from entrance reader'
        pbar=tqdm(total=len(entrance_val),position=0, leave=True,desc=msg)
        for vframe in entrance_val: 
            for readings in df_tracks.iloc[vframe]['RFID_readings']:
                if readings[1] != 'None' and readings[0] ==entrance_reader and vframe>entr_frames and readings[1] in tags:
                    df_tracks=entrance_RFID_match(df_tracks,vframe,readings,RFID_coords,entr_frames,entrance_reader,ent_thres,folder_path)
            pbar.update(1)
    msg='Starting to match RFID readings from Cage readers'
    pbar=tqdm(total=len(cage_val),position=0, leave=True,desc=msg)
    for vframe in cage_val:
        for readings in df_tracks.iloc[vframe]['RFID_readings']:
             if readings[1] != 'None' and readings[0] !=entrance_reader and readings[1] in tags:
                 df_tracks=cage_RFID_match(df_tracks,vframe,readings,RFID_coords,ent_thres,correct_iou,\
                                           reader_thresh,RFID_dist,entrance_reader,folder_path)
        pbar.update(1)
    df_tracks['RFID_tracks']= df_tracks['RFID_tracks'].map(lambda x: duplicate_remove(x))
    df_tracks['lost_tracks']=df_tracks.apply(lambda x:get_lost_tracks(x['sort_tracks'],x['RFID_tracks']),axis=1)
    df_tracks['ID_marked']=[id_tracked(x,y) for x,y in zip (df_tracks.sort_tracks.values,df_tracks.RFID_tracks.values)]
    return df_tracks,validation_frames

def sort_track_list(track_list):
    if track_list ==[]:
        return []
    else:
        track_list=track_list.sort(key=lambda x: x[4])
        return track_list



def remove_match(df_tracks,frame,sort_id,rfid,correct_iou,ent_thres,entrance_reader,RFID_coords,readout):
    sort_track=[strack for strack in df_tracks.iloc[frame]['sort_tracks']if strack[4] ==sort_id][0]#the sort id in question
    rfid_track_1=[rtracks for rtracks in df_tracks.iloc[frame]['RFID_tracks'] if rtracks[:4]==sort_track[:4]]# the RFID markedt ont he sort_id
    rfid_track_2=[rtracks for rtracks in df_tracks.iloc[frame]['RFID_tracks'] if rtracks[4]==readout[1]] #the correct RFID on an other sort_id
    sided=None
    if len(rfid_track_1)>0 and len(rfid_track_2) ==0:
        index_checklist_f=[ind for ind,sids in enumerate(df_tracks.iloc[frame:]['track_id'].values) if sort_id in sids]
        index_checklist_b=list(reversed([ind for ind,sids in enumerate(df_tracks.iloc[:frame]['track_id'].values) if sort_id in sids]))
        df_tracks=remove_match_forward(df_tracks,frame,sort_id,index_checklist_f)
        df_tracks,frame2stopmatch=remove_match_backward(df_tracks,sort_id,entrance_reader,RFID_coords,correct_iou,ent_thres,index_checklist_b,frame)
        id_correted={sort_id:frame2stopmatch,sided:None}
        return df_tracks,frame2stopmatch,id_correted
    elif len(rfid_track_1)==0 and len(rfid_track_2) >0:
        sided=[strack for strack in df_tracks.iloc[frame]['sort_tracks'] if strack[:4]==rfid_track_2[0][:4]][0][4]
        index_checklist_f=[ind for ind,sids in enumerate(df_tracks.iloc[frame:]['track_id'].values) if sided in sids]
        index_checklist_b=list(reversed([ind for ind,sids in enumerate(df_tracks.iloc[:frame]['track_id'].values) if sided in sids]))
        df_tracks=remove_match_forward(df_tracks,frame,sided,index_checklist_f)
        #print(frame)
        df_tracks,frame2stopmatch=remove_match_backward(df_tracks,sided,entrance_reader,RFID_coords,correct_iou,ent_thres,index_checklist_b,frame)
        id_correted={sort_id:None,sided:frame2stopmatch}
        return df_tracks,frame2stopmatch,id_correted
    elif len(rfid_track_1)>0 and len(rfid_track_2) >0:
        sided=[strack for strack in df_tracks.iloc[frame]['sort_tracks'] if strack[:4]==rfid_track_2[0][:4]][0][4]
        index_checklist_f1=[ind for ind,sids in enumerate(df_tracks.iloc[frame:]['track_id'].values) if sided in sids]
        index_checklist_b1=list(reversed([ind for ind,sids in enumerate(df_tracks.iloc[:frame]['track_id'].values) if sided in sids]))
        df_tracks,frame2stopmatch1=remove_match_backward(df_tracks,sided,entrance_reader,RFID_coords,correct_iou,ent_thres,index_checklist_b1,frame)
        index_checklist_f2=[ind for ind,sids in enumerate(df_tracks.iloc[frame:]['track_id'].values) if sort_id in sids]
        index_checklist_b2=list(reversed([ind for ind,sids in enumerate(df_tracks.iloc[:frame]['track_id'].values) if sort_id in sids]))
        df_tracks=remove_match_forward(df_tracks,frame,sort_id,index_checklist_f2)
        df_tracks=remove_match_forward(df_tracks,frame,sided,index_checklist_f1)
        df_tracks,frame2stopmatch2=remove_match_backward(df_tracks,sort_id,entrance_reader,RFID_coords,correct_iou,ent_thres,index_checklist_b2,frame)
        if frame2stopmatch1 != frame2stopmatch2:
            sort_id_list=[sort_id,frame2stopmatch2,index_checklist_b2]
            sided_list=[sided,frame2stopmatch1,index_checklist_b1]
            if frame2stopmatch1 ==frame or frame2stopmatch2 ==frame:
                frame2stopmatch=min(frame2stopmatch1,frame2stopmatch2)
            else:
                df_tracks,frame2stopmatch=remove_match_revised(df_tracks,sort_id_list,sided_list,correct_iou,entrance_reader,ent_thres,RFID_coords)
        else:
            frame2stopmatch=frame2stopmatch1
        id_correted={sort_id:frame2stopmatch,sided:frame2stopmatch}
        return df_tracks,frame2stopmatch,id_correted
    else:
        print('unknow correction pattern at {frame}')
        import sys
        sys.exit()


def cage_RFID_match(df_tracks,frame,readout,RFID_coords,ent_thres,correct_iou,reader_thresh,RFID_dist,entrance_reader,folder_path):
    track_distances=[distance_box_RFID(readout[0],k,RFID_coords) for k in df_tracks.iloc[frame]['sort_tracks']]
    track_distances_r=track_distances
    track_distances= {t:k for t,k in enumerate(track_distances) if k<RFID_dist} # gets the tracks within a certain range of an RFID reader
    track_iou_RFID=[RFID_ious(readout[0],t,RFID_coords) for t in df_tracks.iloc[frame]['sort_tracks']] 
    track_iou_RFID_r=track_iou_RFID
    track_iou_RFID={t:k for t,k in enumerate(track_iou_RFID) if k > reader_thresh} 
    tracks_exist=[t for t in df_tracks.iloc[frame]['RFID_tracks'] if t[4] ==readout[1]] # sees if the RFID is already matched
    #two cases: 1: a tracks already contains the RFID, 2 the current track is already marked 3. the above 2 situations
    path=folder_path+'/matching_process_cage.csv'
    with open(path,'a') as file:
         file.write(f'"{str(frame)}","{str(readout[0])}","{str(readout[1])}","{str(RFID_coords[readout[0]])}","{str(df_tracks.iloc[frame]["sort_tracks"])}","{str(track_distances_r)}","{str(track_iou_RFID_r)}","{str(tracks_exist)}"\n')
    if len(track_distances) ==1 and len(track_iou_RFID)==1 and len(tracks_exist) ==0:
        index=max(track_iou_RFID,key=lambda k:track_iou_RFID[k])
        sort_id=df_tracks.iloc[frame]['sort_tracks'][index][4]
        start_match=True
        if df_tracks.iloc[frame]['sort_tracks'][index][:4] in [rtracks[:4] for rtracks in df_tracks.iloc[frame]['RFID_tracks']]:
            reid=True
        else:
            reid=False
    elif len(track_distances) ==1 and len(track_iou_RFID)== 1 and len(tracks_exist) ==1:
        index=max(track_iou_RFID,key=lambda k:track_iou_RFID[k])
        sort_id=df_tracks.iloc[frame]['sort_tracks'][index][4]
        if df_tracks.iloc[frame]['sort_tracks'][index][:4] ==tracks_exist[0][:4]:
            start_match=False
            reid=False
            details=[f'SID {sort_id} already matched to tag',readout[1]]
            df_tracks.iloc[frame]['Matching_details']+=[details]
        else:
            reid=True
            start_match=True
    else:
        start_match=False
        reid=False
        details=[f'Matching conditions not met for reader {readout[0]}',readout[1]]
        df_tracks.iloc[frame]['Matching_details']+=[details]
    if  start_match:
        matched_id=[sort_id,readout[1],readout[0]]
        df_tracks.iloc[frame]['RFID_matched']+= [matched_id]
        max_f_match=[ind+frame for ind,sids in enumerate(df_tracks.iloc[frame:]['track_id'].values) if sort_id in sids]
        max_b_match=list(reversed([ind for ind,sids in enumerate(df_tracks.iloc[:frame]['track_id'].values) if sort_id in sids]))
        forward_rframe=None
        if reid:
            df_tracks,backward_rframe,corrections=remove_match(df_tracks,frame,sort_id,readout[1],correct_iou,ent_thres,entrance_reader,RFID_coords,readout)
            df_tracks.iloc[frame]['Correction']+=[['on',backward_rframe,frame,corrections]]
        else:
            backward_rframe=None
        df_tracks=RFID_SID_match(sort_id,max_b_match,backward_rframe,df_tracks,readout,entrance_reader,RFID_coords,ent_thres,'backward',False,frame)
        df_tracks=RFID_SID_match(sort_id,max_f_match,forward_rframe,df_tracks,readout,entrance_reader,RFID_coords,ent_thres,'forward',False,frame)
    else:
        pass
    return df_tracks

def entrance_RFID_match(df,frame,readout,RFID_coords,entr_frames,entrance_reader,ent_thres,folder_path):
    if frame<len(df)-1:
        dets=max(df.iloc[frame-entr_frames:frame+entr_frames]['sort_entrance_dets'])
        dets_b=max(df.iloc[frame-entr_frames:frame-1]['sort_entrance_dets'])
        dets_f=max(df.iloc[frame+1:frame+entr_frames]['sort_entrance_dets'])
        dets_b_ids=set([t for k in df.iloc[frame-entr_frames:frame]['Entrance_ids'] for t in k if k !=[]])
        dets_f_ids=set([t for k in df.iloc[frame+1:frame+entr_frames]['Entrance_ids'] for t in k if k !=[]])
        path=folder_path+'/matching_process_entrance.csv'
        with open(path,'a') as file:
            file.write(f'"{str(frame)}","{str(readout[1])}","{str(list(dets_b_ids))}","{str(list(dets_f_ids))}"\n')
        if dets>1:
            details=['Tag read at entrance but no match',readout[1]]
            df.iloc[frame]['Matching_details']+=[details]
        elif dets_b ==1 and dets_f==1:
            details=['Tag read at entrance but no match',readout[1]]
            df.iloc[frame]['Matching_details']+=[details]
        elif dets_b ==0 and dets_f ==1 and 0<len(dets_f_ids)<2:
            index_n=df.iloc[frame+1:frame+entr_frames].query('sort_entrance_dets ==1').index[0]
            sort_id_entrance=df.iloc[index_n]['Entrance_ids'][0]
            max_f_match=[ind+frame for ind,sids in enumerate(df.iloc[index_n:]['track_id'].values) if sort_id_entrance in sids]
            tracks_exist=[t for t in df.iloc[index_n]['RFID_tracks'] if t[4] ==readout[1]]
            frames_iou=[value for ious_dic in df.iloc[index_n-5:index_n+5]['ious'].values for key,value in ious_dic.items() \
                        if sort_id_entrance in key and value>0]
            if len(tracks_exist)==0 and len(frames_iou)==0:
                forward_rframe=None
                matched=[[sort_id_entrance,'Tag Entered Cage',readout[1]]] 
                df.iloc[index_n]['RFID_matched']+= matched
                df=RFID_SID_match(sort_id_entrance,max_f_match,forward_rframe,
                                  df,readout,entrance_reader,RFID_coords,ent_thres,'forward',True,frame)
        elif dets_f ==0 and dets_b ==1 and 0<len(dets_b_ids)<2:
            index_n=df.iloc[frame-entr_frames:frame-1].query('sort_entrance_dets ==1').index[-1]
            sort_id_entrance=df.iloc[index_n]['Entrance_ids'][0]
            tracks_exist=[t for t in df.iloc[index_n]['RFID_tracks'] if t[4] ==readout[1]]
            max_b_match=list(reversed([ind for ind,sids in enumerate(df.iloc[:index_n-1]['track_id'].values) if sort_id_entrance in sids]))
            frames_iou=[value for ious_dic in df.iloc[index_n-5:index_n+5]['ious'].values for key,value in ious_dic.items() \
                        if sort_id_entrance in key and value>0]
            if len(tracks_exist)==0:
                backward_rframe=None
                matched=[[sort_id_entrance,'Tag Existed Cage',readout[1]]] 
                df.iloc[index_n]['RFID_matched']+= matched
                df=RFID_SID_match(sort_id_entrance,max_b_match,backward_rframe,
                                  df,readout,entrance_reader,RFID_coords,ent_thres,'backward',True,frame)
        else:
            details=['Tag read at entrance but no match',readout[1]]
            df.iloc[frame]['Matching_details']+=[details]
    else:
        pass
    return df

def label_correct_multi(df_tracks_out):
    ctemp=[]
    correct_RFIDtracks=[i for i in df_tracks_out[df_tracks_out.Tracked_marked.map(len)>0].index]
    for ind in correct_RFIDtracks:
        if len(df_tracks_out.iloc[ind]['Tracked_marked']) == 1:
            corrected=False
            for i in df_tracks_out.iloc[ind]['Tracked_marked'].values():
                for v in i:
                    if not corrected:
                        RFID_2check=df_tracks_out.iloc[ind]['RFID_tracks'][v][4]
                        RFID_list=[tracks[4] for tracks in df_tracks_out.iloc[ind]['RFID_tracks']]
                        RFID_dup=[index for index,RFID in enumerate(RFID_list) if RFID==RFID_2check]
                        if len(RFID_dup)>1:
                            df_tracks_out.at[ind,'RFID_tracks']=[tracks for index, tracks in enumerate(df_tracks_out.iloc[ind]['RFID_tracks']) if index !=v]
                            corrected=True
                        else: 
                            pass
            if corrected:
                ctemp.append(ind)
            else:
                df_tracks_out.at[ind,'RFID_tracks']=[tracks for index, tracks in enumerate(df_tracks_out.iloc[ind]['RFID_tracks']) if index not in i]
        else:
            pass
    correct_RFIDtracks=[i for i in correct_RFIDtracks if i not in ctemp]
    df_tracks_out['lost_tracks']=df_tracks_out.apply(lambda x:get_lost_tracks(x['sort_tracks'],x['RFID_tracks']),axis=1)
    return df_tracks_out,correct_RFIDtracks


def get_lost_tracks(det,RFID_tracked):
    tracked=[i[:3] for i in RFID_tracked]
    lost=[i for i in det if i[:3] not in tracked]
    return lost


"""
def cleanup(df_tracks_out):
    df_tracks_out['RFID_tracks']=[delete_mutilbaled(x) for x in df_tracks_out['RFID_tracks'].values]
    df_tracks_out['RFID_tracks']=[delete_mutilbaled2(x) for x in df_tracks_out['RFID_tracks'].values]
    df_tracks_out['lost_tracks']=df_tracks_out.apply(lambda x:get_lost_tracks(x['sort_tracks'],x['RFID_tracks']),axis=1)
    return df_tracks_out
"""


def tag_left_recover_simp(df,tags):
    list_to_check_final=get_left_over_tag_indexes(df,tags)
    list_to_check_final=[z for z in list_to_check_final if len(df.iloc[z].lost_tracks) ==1]
    if len(list_to_check_final) !=0:
        read_length=len(list_to_check_final)
        pbar = tqdm(total=read_length,position=0, leave=True,desc='Matching last tag in frame')
        for i in list_to_check_final:
            RFID_list_i=[z[4] for z in df.iloc[i].RFID_tracks]
            if len(set(RFID_list_i)) == len(tags):
                pass
            else:
                tag_left=list(set(tags)-set(RFID_list_i))[0]
                #unmatched_id=df.iloc[i].lost_tracks[0][4]
                tracked_left=df.iloc[i].lost_tracks[0][:4]+[tag_left]
                track_f= df.iloc[i]['RFID_tracks']+[tracked_left]
                df.at[i,'RFID_tracks']=track_f
                matched_id=[df.iloc[i].lost_tracks[0][4],tag_left,'Leftover Matched']
                df.iloc[i]['RFID_matched']+= [matched_id]
            pbar.update(1)
    df['lost_tracks']=df.apply(lambda x:get_lost_tracks(x['sort_tracks'],x['RFID_tracks']),axis=1)
    return df

def rfid2bpts(bpts,RFIDs,slack,bpt2look=['snout','tail_base']):
    bpts_marked=[]
    bpt_interest=[i for i in bpts if i[2] in bpt2look]
    for bb in RFIDs:
        rfid_bpt=[]
        for bpt in bpt_interest:
            if  bboxContains(bb,bpt,slack):
                rfid_bpt.append(bpt+[bb[4]])
        bpts_marked.append(rfid_bpt)
    for i in range(len(bpts_marked)):
        for y in bpts_marked[i]:
            inds=[ind for ind,val in enumerate(bpts_marked[i]) if y[:3] in [val_list[:3] for val_list in bpts_marked[i]]]
            if len(inds)>1:
                dbpt=y[2]
                ins=[bpt for bpt in bpts_marked[i] if bpt[2]==dbpt and bpt !=y]
                if len(ins)>0:
                    bpts_marked[i].remove(y)     
    flat_list=[i[:3] for y in bpts_marked for i in y]
    ele2remove=[i for i in flat_list if flat_list.count(i)>1]
    ele2remove=[v for i,v in enumerate(ele2remove) if v not in ele2remove[:i]]
    for i in range(len(bpts_marked)):
        bpts_marked[i]=[y for y in bpts_marked[i] if y[:3] not in ele2remove]
    for c in range(len(bpts_marked)):
        for y in [i[2] for i in bpts_marked[c]]:
            inds=[ind for ind, val in enumerate([i[2] for i in bpts_marked[c]]) if val==y]
            if len(inds)>1:
                bpts_marked[c]=[v for ind,v in enumerate(bpts_marked[c]) if ind not in inds]
    flat_list=[i[:3] for y in bpts_marked for i in y]
    undetemined_bpt=[i for i in bpt_interest if i not in flat_list]
    return bpts_marked, undetemined_bpt




def bpt_distance_compute(row,tags,bpt_int):
    dic2return=[]
    if len(tags)>1:
        for combi in itertools.permutations(tags, 2):
            dic={}
            for animal_bpt in row[str(combi[0])+'_bpts']:
                if animal_bpt[2] in bpt_int:
                    for animal_bpt2 in row[str(combi[1])+'_bpts']:
                        part=animal_bpt[2]+'-'+animal_bpt2[2]
                        Dist= Distance(animal_bpt,animal_bpt2)
                        dic[part]=[animal_bpt[0],animal_bpt[1],animal_bpt2[0],animal_bpt2[1],Dist]
            dic2return.append(dic)
    else:
        dic2return=[]
    return dic2return

def get_tracked_activity(motion_status,motion_roi,RFID_tracks,tags):
    actives={}
    if motion_status == 'Motion':    
        for tracked in RFID_tracks:
            active= 'Non-active'
            for rois in motion_roi:
                intersect=iou(tracked,rois)
                if intersect >0:
                    active='Active'
            actives[tracked[4]]=active
    else:
        for tracked in RFID_tracks:
            actives[tracked[4]]='Non_active'
    for RFID in tags:
        if RFID not in actives.keys():
            actives[RFID]='Not_tracked'
    return actives



def bbox_revised(df_tracks,df_RFID,RFID_coords,entrance_reader,thresh,threshold,ent_thres,n_mice,pathin,area_thres,resolution):
    vid_path=pathin+'/raw.avi'
    if not os.path.exists(vid_path):
        vid_path=pathin+'/raw.mp4'
    combined_df=Combine_RFIDreads(df_tracks,df_RFID)
    if n_mice !=1:
        validation_frames=combined_df[combined_df.RFID_readings.notnull()].index
    else:
        validation_frames=[]
    if entrance_reader is None:
        entrance_val=[]
    else:
        entrance_val=list(set([i for i in validation_frames for x in combined_df.iloc[i]['RFID_readings'] if x[0]==entrance_reader]))
    entrance_times=[combined_df.iloc[t]['Time'] for t in entrance_val]
    entrance_times.sort()
    entrance_times=np.asarray(entrance_times)
    ent_count = [0]
    time_list=combined_df['Time'].values
    tracks=combined_df['sort_tracks'].values
    tracks_pred=combined_df['unmatched_predicts'].values
    msg='Checking frames for occlusions/yolov4 fails'
    pbar= tqdm(total=len(combined_df),position=0,leave=True,desc=msg)
    frames2check=[]
    for i in range(len(combined_df)):
        if i!=0:
            if tracks[i].shape[0] < tracks[i-1].shape[0]:
                if True:
                    if len(tracks[i]) ==0:
                        current_ids=np.asarray([])
                    else:
                        try:
                            current_ids=tracks[i][:,4]
                        except Exception:
                            #print(tracks[i])
                            #print(type(tracks[i]))
                            import sys
                            sys.exit()
                    if len(tracks[i-1]) ==0:
                        prev_ids=np.asarray([])
                    else:    
                        prev_ids=tracks[i-1][:,4]
                    pass
                    missing_bb=[strack for strack in tracks[i-1] if strack[4] in prev_ids and strack[4] not in current_ids]
                    for strack_inde,strack in enumerate(missing_bb):
                        iteraction_list=[iteraction(strack,strack_prev) for strack_prev in  tracks[i-1]]
                        iteraction_list=[it_val for it_val in iteraction_list if it_val>threshold]
                        if iteraction_list ==[]:
                            missing_bb.pop(strack_inde)
                    if len(missing_bb)>0:
                        for strack in tracks_pred[i]:
                            iou_list=[iou(strack, strack_m) for strack_m in missing_bb]
                            iou_list=[inde_val for inde_val, iou_val in enumerate(iou_list) if iou_val>0.5]
                            if len(iou_list)>0:
                                    if len(tracks[i])+1<= n_mice:
                                        if entrance_reader is None:
                                            tracks_add=np.append(strack[:4],int(missing_bb[max(iou_list)][4]))
                                            tracks[i]=np.append(tracks[i],[tracks_add],axis=0)
                                            frames2check.append([i,tracks_add.tolist()])
                                            missing_bb.pop(max(iou_list))
                                        else:
                                            if distance_to_entrance(strack,RFID_coords,entrance_reader) >ent_thres+20:
                                                tracks_add=np.append(strack[:4],int(missing_bb[max(iou_list)][4]))
                                                tracks[i]=np.append(tracks[i],[tracks_add],axis=0)
                                                frames2check.append([i,tracks_add.tolist()])
                                                missing_bb.pop(max(iou_list))
                                    #####use in second pipeline
                                    else:#just uses previous frame's predict for it 
                                        if entrance_reader is None:
                                            if len(tracks[i])+1<= n_mice:
                                                track_add= [strack2 for strack2 in missing_bb if strack2[4]==int(missing_bb[max(iou_list)][4])][0]
                                                #frames2check.append([i,track_add,track_add[4]])
                                                tracks[i]=np.append(tracks[i],[tracks_add],axis=0)
                                                frames2check.append([i,tracks_add.tolist()])
                                                missing_bb.pop(max(iou_list))
                                        else:
                                            if len(tracks[i])+1<= n_mice:
                                                if distance_to_entrance(strack,RFID_coords,entrance_reader) >ent_thres+20:
                                                    track_add= [strack2 for strack2 in missing_bb if strack2[4]==int(missing_bb[max(iou_list)][4])][0]
                                                    tracks[i]=np.append(tracks[i],[tracks_add],axis=0)
                                                    frames2check.append([i,tracks_add.tolist()])
                                                    missing_bb.pop(max(iou_list))
        pbar.update(1)
    tracks=[array2list(t) for t in tracks]

    msg='Checking Kalmen filter predictions at frames with occlusions/yolov4 fails'
    pbar= tqdm(total=len(frames2check),position=0,leave=True,desc=msg)
    frames2extract=list(set([i[0] for i in frames2check]))
    if len(frames2extract)>1000:
        extract_sublists=list_split(frames2extract,1000)
        for frame_sublist in extract_sublists:
            vr = VideoReader(vid_path, ctx=cpu(0))
            frames = vr.get_batch(list(frame_sublist)).asnumpy()
            frame_check=[bb_predicts for bb_predicts in frames2check if bb_predicts[0] in frame_sublist]
            for bbs in frame_check:
                frame_inde= list(frame_sublist).index(bbs[0])
                missing_bb=[strack for strack in tracks[i-1] if strack[4] == bbs[1][4]]
                pred_cent=bbox_to_centroid(bbs[1])
                if not bb_contain_mice_check(frames[frame_inde],bbs[1],30):
                    temp_list=tracks[bbs[0]]
                    temp_list=[strack for strack in temp_list if strack != bbs[1]]
                    if len(missing_bb) !=0 :
                        temp_list.append(missing_bb[0])
                    tracks[bbs[0]]=temp_list
                elif bbox_area(bbs[1])<area_thres[0] or bbox_area(bbs[1])>area_thres[1]:
                    temp_list=tracks[bbs[0]]
                    temp_list=[strack for strack in temp_list if strack != bbs[1]]
                    if len(missing_bb) !=0 :
                        temp_list.append(missing_bb[0].tolist())
                    tracks[bbs[0]]=temp_list
                elif pred_cent[0]>resolution[0] or pred_cent[0]<0 or pred_cent[1]>resolution[1] or pred_cent[1]<0:
                    temp_list=tracks[bbs[0]]
                    temp_list=[strack for strack in temp_list if strack != bbs[1]]
                    if len(missing_bb) !=0 :
                        temp_list.append(missing_bb[0])
                    tracks[bbs[0]]=temp_list
                pbar.update(1)
    else:
        vr = VideoReader(vid_path, ctx=cpu(0))
        frames = vr.get_batch(list(frames2extract)).asnumpy()
        for bbs in frames2check:
            frame_inde= list(frames2extract).index(bbs[0])
            missing_bb=[strack for strack in tracks[i-1] if strack[4] == bbs[1][4]]
            pred_cent=bbox_to_centroid(bbs[1])
            if not bb_contain_mice_check(frames[frame_inde],bbs[1],30):
                temp_list=tracks[bbs[0]]
                temp_list=[strack for strack in temp_list if strack != bbs[1]]
                if len(missing_bb) !=0 :
                    temp_list.append(missing_bb[0])
                tracks[bbs[0]]=temp_list
            elif bbox_area(bbs[1])<area_thres[0] or bbox_area(bbs[1])>area_thres[1]:
                temp_list=tracks[bbs[0]]
                temp_list=[strack for strack in temp_list if strack != bbs[1]]
                if len(missing_bb) !=0 :
                    temp_list.append(missing_bb[0])
                tracks[bbs[0]]=temp_list
            elif pred_cent[0]>resolution[0] or pred_cent[0]<0 or pred_cent[1]>resolution[1] or pred_cent[1]<0:
                temp_list=tracks[bbs[0]]
                temp_list=[strack for strack in temp_list if strack != bbs[1]]
                if len(missing_bb) !=0 :
                    temp_list.append(missing_bb[0])
                tracks[bbs[0]]=temp_list
            pbar.update(1)
    combined_df['sort_tracks'] = tracks
    if n_mice != 1:
        combined_df=combined_df.drop(columns=['Time','RFID_readings'])
    else: 
        combined_df=combined_df.drop(columns=['Time'])
    return combined_df

def meet_criteria_trigger(tracks,tracks_prev,i,ent_count,time_threshs):
    if tracks_prev.shape[0] - tracks.shape[0] > 1:
        return True#True # more than 1 disappear. We assume that only one mouse can exit from tunnel at time T.
    else:
        if len(np.where(np.absolute(time_threshs)<1)[0])>0: # if there is a mouse enter tunnel and it was detected within x second。
            return True #False #  Mouse number -1 is normal, Pass,
        else:  # check it is mouse do not enter or do not detected by RFID
            if i <=3: # information is not enough
                return False #False # Pass
            else:
                
                if ent_count[0] > 0 and ent_count[1] > 0:
                    if ent_count[1] - ent_count[2] ==1:
                        return False # False # preivious two frames there are N mice which are close to entrance,
                        #but at current frame, N-1 are close enough to entrance, it means one exit the cage via entrance,Pass
                    else:
                        return True#True # not the mice around entrance disappear, trigger
                else:
                    return True#True


def id_tracked(sort_tracks,RFID_tracks):
    if len(RFID_tracks) ==0:
        return {}
    else:
        m_dic={}
        for track in sort_tracks:
            t_matched=[t for t in RFID_tracks if t[:4]==track[:4]]
            if len(t_matched) != 0:
                m_dic[track[4]]=t_matched[0][4]
        return m_dic


def remove_match_forward(df_tracks,frame,sort_id,index_checklist_f):
    #can optimize
    for inde in index_checklist_f:
        sid_track=[sid for sid in df_tracks.iloc[frame+inde]['sort_tracks'] if sid[4]==sort_id][0]
        track_rf=[t for t in df_tracks.iloc[frame+inde]['RFID_tracks'] if t[:4] !=sid_track[:4]]
        df_tracks.at[frame+inde,'RFID_tracks']=track_rf
    return df_tracks



def remove_match_backward(df_tracks,sort_id,entrance_reader,RFID_coords,correct_iou,ent_thres,index_checklist_b,frame):
    backward_remove=True
    if len(index_checklist_b):
        for inde in index_checklist_b:
            if backward_remove:
                if sort_id in df_tracks.iloc[inde]['track_id']:
                    iou_dic_b={i:v  for i,v in df_tracks.iloc[inde]['ious'].items() if sort_id in i}
                    track2check=[strack for strack in df_tracks.iloc[inde]['sort_tracks'] if strack[4] ==sort_id][0]
                    if entrance_reader is None:
                        if len(iou_dic_b) != 0:
                            if len({i:v  for i,v in iou_dic_b.items() if v>correct_iou})>0:
                                track_rf=[t for t in df_tracks.iloc[inde]['RFID_tracks'] if  t[:4] !=track2check[:4]]
                                df_tracks.at[inde,'RFID_tracks']=track_rf
                                backward_remove=False
                            else:
                                track_rf=[t for t in df_tracks.iloc[inde]['RFID_tracks'] if  t[:4] !=track2check[:4]]
                                df_tracks.at[inde,'RFID_tracks']=track_rf
                        else:
                            track_rf=[t for t in df_tracks.iloc[inde]['RFID_tracks'] if  t[:4] !=track2check[:4]]
                            df_tracks.at[inde,'RFID_tracks']=track_rf
                    else: 
                        if distance_to_entrance(track2check,RFID_coords,entrance_reader) >ent_thres:
                            #print(distance_to_entrance(track2check,RFID_coords,entrance_reader))
                            #print(ent_thres)
                            if len(iou_dic_b) != 0:
                                if len({i:v  for i,v in iou_dic_b.items() if v>correct_iou})>0:
                                    backward_remove=False
                                else:
                                    track_rf=[t for t in df_tracks.iloc[inde]['RFID_tracks'] if  t[:4] !=track2check[:4]]
                                    df_tracks.at[inde,'RFID_tracks']=track_rf
                            else:
                                track_rf=[t for t in df_tracks.iloc[inde]['RFID_tracks'] if  t[:4] !=track2check[:4]]
                                df_tracks.at[inde,'RFID_tracks']=track_rf
                        else:
                            backward_remove=False
            else:
                break
            frame2stopmatch=inde
        return df_tracks,frame2stopmatch
    else:
        return df_tracks,frame

def remove_match_revised(df_tracks,sort_id_list,sided_list,correct_iou,entrance_reader,ent_thres,RFID_coords):
    #finds out which revises frame in smaller
    if sort_id_list[1]<sided_list[1]:
        df_tracks,frame2stop=remove_match_revised2(df_tracks,sided_list,sort_id_list,correct_iou,entrance_reader,
                                                   ent_thres,RFID_coords)
    elif sort_id_list[1]>sided_list[1]:
        df_tracks,frame2stop=remove_match_revised2(df_tracks,sort_id_list,sided_list,correct_iou,entrance_reader,
                                                   ent_thres,RFID_coords)
    return df_tracks,frame2stop
    
def remove_match_revised2(df_tracks,sort_id_list,sided_list,correct_iou,entrance_reader,ent_thres,RFID_coords):
    #get the true frame to revised/correct until
    #sort_id_list has the higher frame2stop at 
    # check if at frame~= sided frane2stop iou 
    if len(sort_id_list[2])!=0:
        frame2check=min(sort_id_list[2], key=lambda x:abs(x-sided_list[1]))
        frame_iou=[ind for ind,iou in df_tracks.iloc[frame2check]['ious'].items() 
                   if sort_id_list[0] in ind and iou>correct_iou]
        #nothing > correct iou
        if len(frame_iou)==0:
            revise=True
        else:
            #can further trace back
            revise=False
    else:
        revise=False
    if revise:
       #try:
       if frame2check <min(sort_id_list[2]):
           index_iou_check=sort_id_list[2]
       else:
           index_iou_check=sort_id_list[2][sort_id_list[2].index(frame2check):]
       #except Exception as e:
       #    print(e)
       #    print(sort_id_list)
       #    print(sided_list)
       #index_iou_check=sort_id_list[2][sort_id_list[2].index(sided_list[1])-1:]
       for inde in index_iou_check:
           track2check=[strack for strack in df_tracks.iloc[inde]['sort_tracks']
                        if strack[4] ==sort_id_list[0]][0]
           iou_dic_b={i:v  for i,v in df_tracks.iloc[inde]['ious'].items() if sort_id_list[0] in i}
           if entrance_reader is None:
               if len(iou_dic_b) != 0:
                   if len({i:v  for i,v in iou_dic_b.items() if v>correct_iou})>0:
                       break
           else:
               if distance_to_entrance(track2check,RFID_coords,entrance_reader) >ent_thres:
                   break
               if len(iou_dic_b) != 0:
                   if len({i:v  for i,v in iou_dic_b.items() if v>correct_iou})>0:
                       break
       try:
            frame2stop=inde
       except Exception as e:
            print(index_iou_check)
            print(frame2check)
            print(sort_id_list)
            print(sided_list)
            import sys
            sys.exit()
    else:
       frame2stop=sided_list[1]
    if frame2stop<sided_list[1]:
       sided_list[2]=sided_list[2][sided_list[2].index(sided_list[1])-1:]
       for f in sided_list[2]:
            if f>=frame2stop:
                track2check=[strack for strack in df_tracks.iloc[f]['sort_tracks'] if strack[4] ==sided_list[0]][0]
                track_rf=[t for t in df_tracks.iloc[f]['RFID_tracks'] if  t[:4] !=track2check[:4]]
                df_tracks.at[f,'RFID_tracks']=track_rf
            else:
                break
    sort_id_list[2]=sort_id_list[2][sort_id_list[2].index(sort_id_list[1])-1:]
        #print('saved')
    for f in sort_id_list[2]:
        if f>=frame2stop:
            track2check=[strack for strack in df_tracks.iloc[f]['sort_tracks'] if strack[4] ==sort_id_list[0]][0]
            track_rf=[t for t in df_tracks.iloc[f]['RFID_tracks'] if  t[:4] !=track2check[:4]]
            df_tracks.at[f,'RFID_tracks']=track_rf
        else:
            break
    return df_tracks, frame2stop



def RFID_SID_match(sort_id,match_frames,stop_frame,df_tracks,
                   readout,entrance_reader,RFID_coords,
                   ent_thres,direction,ent,frame,match_length=5):
    #speed up?
    #needs to label sid 5 >frames
    #stops if the rfid/sid is marked
    #df_tracks_temp=df_tracks.copy(deep=True)
    if ent:
        entrance_reader=None
    else:
        pass
    match=True
    df_tracks_temp=df_tracks.copy(deep=True)
    count=0
    for inde in match_frames:
        if direction == 'forward':
            match_start = (stop_frame is None or (stop_frame is not None and inde<stop_frame))
        elif direction == 'backward':
            match_start = (stop_frame is None or (stop_frame is not None and inde>=stop_frame))
        if match:
            if match_start:
                if sort_id in df_tracks_temp.iloc[inde]['track_id']:
                    track_matched=[k for k in df_tracks_temp.iloc[inde]['sort_tracks'] if k[4]==sort_id][0]
                    track=[[track_matched[0],track_matched[1],track_matched[2],track_matched[3],readout[1]]]
                    rfid_labeled=[t for t in df_tracks_temp.iloc[inde]['RFID_tracks'] if t[:4] == track[0][:4] or t[4]==readout[1]]
                    if len(rfid_labeled) ==0: 
                        if entrance_reader is None:
                            track_f=[t for t in df_tracks_temp.iloc[inde]['RFID_tracks'] ]+track
                            df_tracks_temp.at[inde,'RFID_tracks']=track_f
                            count+=1
                        else:
                            if distance_to_entrance(track[0],RFID_coords,entrance_reader) >ent_thres:
                                track_f=[t for t in df_tracks_temp.iloc[inde]['RFID_tracks']]+track
                                df_tracks_temp.at[inde,'RFID_tracks']=track_f
                                count+=1
                            else:
                                track_f=[t for t in df_tracks_temp.iloc[inde]['RFID_tracks']]+track
                                df_tracks_temp.at[inde,'RFID_tracks']=track_f
                                count+=1
                                match=False
                    else:
                        match=False
                        break
            else:
                match=False
                break
        else:
            break
    if count>match_length:
        df_tracks=df_tracks_temp
    return df_tracks


def match_left_over_tag(df,tags,config_dict_analysis):
    entrance_reader=config_dict_analysis['entrance_reader']
    RFID_coords=config_dict_analysis['RFID_readers']
    ent_thres=config_dict_analysis['entrance_distance']
    correct_iou=config_dict_analysis['correct_iou']
    #further optimization
    #break into consectuive sections? 
    frames_done=[]
    loop_count=1
    while True:
        index_list=get_left_over_tag_indexes(df,tags)
        if len(index_list) == 0:
            break
        index_list_refined=refine_frame_left_list(index_list,df)
        msg=f'Starting Left over Tag match comphrensive loop {loop_count}'
        pbar=tqdm(total=len(index_list_refined),position=0,leave=True,desc=msg)
        for i in index_list_refined:
            RFID_tracked=[strack[4] for strack in df.iloc[i].RFID_tracks]
            if len(RFID_tracked)<len(tags):
                tag_left=list(set(tags)-set(RFID_tracked))[0]
                sid_left=df.iloc[i].lost_tracks[0][4]
                # optimize this search section argv?
                index_checklist_f=[ind+i for ind,sids in enumerate(df.iloc[i:]['track_id'].values) if sid_left in sids]#appearance of sid in future frames
                index_checklist_b=list(reversed([ind for ind,sids in enumerate(df.iloc[:i]['track_id'].values) if sid_left in sids]))# appearance of sid in previous frames
                index_checklist_id_marked_f=search_id_marked_list_forward(df,i,sid_left)# appearance of sid being marked by RFID in future frames
                index_checklist_id_marked_b=search_id_marked_list_backward(df,i,sid_left)# appearance of sid being marked by RFID in past frames
                readout=['Remainder',tag_left]
                #backward matching
                if len(index_checklist_id_marked_b) ==0:#not marked before
                    backward_rframe=None
                else:
                    if tag_left==df.iloc[index_checklist_id_marked_b[0]]['ID_marked'][sid_left]:# marked as same RFID as before
                        backward_rframe=index_checklist_id_marked_b[0]
                    else:
                        frame_iou=get_iou_list_b(df,i,sid_left,correct_iou, limit=None)
                        if len(frame_iou) ==0:
                            frame_iou=0
                        else:
                            frame_iou=frame_iou[0]# last time sid had an iou>correct iou
                        if entrance_reader is None:
                            index_check_list_ent_b=False
                        else:
                            index_check_list_ent_b=search_entrance_list_backward(df,i,sid_left)
                            if index_check_list_ent_b ==[]:
                                index_check_list_ent_b=False
                        if frame_iou>=index_checklist_id_marked_b[0]:#before it was RFIDed has iou>correct iou
                            if index_check_list_ent_b:
                                backward_rframe=min(index_check_list_ent_b[0],frame_iou)#checking for entrance area 
                            else:
                                backward_rframe=frame_iou
                        else:
                            index_checklist_rid_marked_b=search_rfid_marked_list_backward(df,i,tag_left) #past frames when RFID being marked 
                            if index_checklist_rid_marked_b ==[]:
                                index_checklist_rid_marked_b=[0]
                            if index_checklist_id_marked_b[0]>index_checklist_rid_marked_b[0]:# the RFID was marked before the sid was marked
                                if index_check_list_ent_b:#checking for entrance area 
                                    if index_check_list_ent_b[0]>index_checklist_id_marked_b[0]:
                                        backward_rframe=index_check_list_ent_b[0]
                                    else:
                                        backward_rframe=index_checklist_id_marked_b[0]
                                else:
                                    backward_rframe=index_checklist_id_marked_b[0]
                            else:
                                backward_rframe=index_checklist_b[0]+1# too much uncertainity, pass matching process
                df=RFID_SID_match(sid_left,index_checklist_b,backward_rframe,
                               df,readout,entrance_reader,RFID_coords,ent_thres,'backward',False,i)
                #forward match
                    #frame_iou=get_iou_thresh_frame(frame_iou,sid_left,df,correct_iou)# last time sid had an iou>correct iou
                if len(index_checklist_id_marked_f) ==0:#not marked before
                    forward_rframe=None
                else:
                    if tag_left==df.iloc[index_checklist_id_marked_f[0]]['ID_marked'][sid_left]:# marked as same RFID as before
                        forward_rframe=index_checklist_id_marked_f[0]
                    else:
                        frame_iou=get_iou_list_f(df,i,sid_left, correct_iou, limit=None)
                        if len(frame_iou)==0:
                            frame_iou=len(df)
                        else:
                            frame_iou=frame_iou[0]
                        if entrance_reader is None:
                            index_check_list_ent_f=False
                        else:
                            index_check_list_ent_f=search_entrance_list_forward(df,i,sid_left)
                            if index_check_list_ent_f ==[]:
                                index_check_list_ent_f = False
                        if frame_iou<=index_checklist_id_marked_f[0]:
                            if index_check_list_ent_f:
                                forward_rframe=max(index_checklist_id_marked_f[0],frame_iou)
                            else:
                                forward_rframe=frame_iou
                        else:
                            index_checklist_rfid_marked_f=search_rfid_marked_list_forward(df,i,tag_left)# future frames of RFID being marked
                            if index_checklist_rfid_marked_f ==[]:
                                index_checklist_rfid_marked_f=[len(df)]
                            if index_checklist_id_marked_f[0]<index_checklist_rfid_marked_f[0]:# the RFID was marked before the sid was marked
                                if index_check_list_ent_f:#checking for entrance area 
                                    if index_check_list_ent_f[0]>index_checklist_id_marked_f[0]:
                                        forward_rframe=index_check_list_ent_f[0]
                                    else:
                                        forward_rframe=index_checklist_id_marked_f[0]
                                else:
                                    forward_rframe=index_checklist_id_marked_f[0]
                            else:
                                forward_rframe=index_checklist_id_marked_f[0]-1# too much uncertainity, pass matching process
                df=RFID_SID_match(sid_left,index_checklist_f,forward_rframe,
                               df,readout,entrance_reader,RFID_coords,ent_thres,'forward',False,i)
            pbar.update(1)
        if loop_count>3:
            frames_done+=index_list.copy()
            index_list=get_left_over_tag_indexes(df,tags)
            index_list=list(set(index_list)-set(frames_done))
        else:
            index_list=get_left_over_tag_indexes(df,tags)
        if len(index_list) ==0:
            break
        
        if len(index_list) ==0:
            break
        loop_count+=1
    df['RFID_tracks']= df['RFID_tracks'].map(lambda x: duplicate_remove(x))
    df['RFID_matched']= df['RFID_matched'].map(lambda x: duplicate_remove(x))
    df['lost_tracks']=df.apply(lambda x:get_lost_tracks(x['sort_tracks'],x['RFID_tracks']),axis=1)
    df['Tracked_marked']=df.apply(lambda x:get_RFID_tracked(x['RFID_tracks']),axis=1)
    return df

def get_left_over_tag_indexes(df,tags):
    index_list=df[df.RFID_tracks.map(len)==len(tags)-1].index
    index_list=[z for z in index_list if len(df.iloc[z].track_id) ==len(tags)]
    return index_list

def get_iou_thresh_frame(list_iou_frames,sid,df_tracks,correct_iou):
    for frame in list_iou_frames:
        ious=[iou for inde, iou in df_tracks.iloc[frame]['ious'].items()
              if  sid in inde and iou > correct_iou]  
        if len(ious)>0:
            break
        
    return frame

def iteraction(bb_test,bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    xx3 = bb_test[2] -bb_test[0]
    xx4 = bb_gt[2] -  bb_gt[0]
    yy3 =  bb_test[3] - bb_test[1]
    yy4 =  bb_gt[3] -  bb_gt[1]
    ratio = wh/min(xx3*yy3,xx4*yy4)
    return ratio



# very repetitive functions, should think of a better way
#and dirty 

def search_id_marked_list_forward(df,frame,sid,limit=None):
    list_id_marked=[]
    count=0
    for ind,sids in enumerate(df.iloc[frame:]['ID_marked'].values):
        if sid in sids.keys():
            list_id_marked.append(ind+frame)
            break
        count+=1
        if count==limit:
            break
    return list_id_marked

def search_id_marked_list_backward(df,frame,sid,limit=None):
    list_id_marked=[]
    count=0
    for ind,sids in enumerate(reversed(df.iloc[:frame]['ID_marked'].values)):
        if sid in sids.keys():
            list_id_marked.append(frame-ind-1)
            break
        count+=1
        if count==limit:
            break
    return list_id_marked

def search_rfid_marked_list_backward(df,frame,tag,limit=None):
    list_id_marked=[]
    count=0
    for ind,tags in enumerate(reversed(df.iloc[:frame]['ID_marked'].values)):
        if tag in tags.values():
            list_id_marked.append(frame-ind-1)
            break
        count+=1
        if count==limit:
            break
    return list_id_marked

def search_rfid_marked_list_forward(df,frame,tag,limit=None):
    list_id_marked=[]
    count=0
    for ind,tags in enumerate(df.iloc[frame:]['ID_marked'].values):
        if tag in tags.values():
            list_id_marked.append(ind+frame)
            break
        count+=1
        if count==limit:
            break
    return list_id_marked



def search_entrance_list_forward(df,frame,sid,limit=None):
    list_id_marked=[]
    count=0
    for ind,sids in enumerate(df.iloc[:frame]['Entrance_ids'].values):
        if sid in sids:
             list_id_marked.append(ind+frame)
             break
        count+=1
        if count == limit:
             break
    return list_id_marked
             

def search_entrance_list_backward(df,frame,sid,limit=None):
    list_id_marked=[]
    count=0
    for ind,sids in enumerate(reversed(df.iloc[:frame]['Entrance_ids'].values)):
        if sid in sids:
            list_id_marked.append(frame-ind-1)
            break
        count+=1
        if count==limit:
            break
    return list_id_marked


def get_iou_list_f(df,frame,sid, correct_iou, limit=None):
    list_id_marked=[]
    count=0
    for ind, iou in enumerate(df.iloc[:frame]['ious'].values):
        if len([t for t,k in iou.items() if sid in t and k>correct_iou])>0:
            list_id_marked.append(frame+ind)
            break
        count+=1
        if count == limit:
            break
    return list_id_marked

def get_iou_list_b(df,frame,sid, correct_iou, limit=None):
    list_id_marked=[]
    count=0
    for ind, iou in enumerate(reversed(df.iloc[frame:]['ious'].values)):
        if len([t for t,k in iou.items() if sid in t and k>correct_iou])>0:
            list_id_marked.append(frame-ind-1)
            break
        count+=1
        if count == limit:
            break
    return list_id_marked


def sid_left_list(frame_list,df):
    """
    Part of refine_frame_left_list
    """
    return [df.iloc[frame].lost_tracks[0][4] for frame in frame_list]


def refine_frame_left_list(index_list,df):
    """
    reduced number of frames to do last tag match
    groups consecutive frames togeather and return the first of each 
    consecutive frames
    """
    from operator import itemgetter
    from itertools import groupby
    groups=[list(map(itemgetter(1), g)) for k, g in groupby(enumerate(index_list), lambda x: x[0]-x[1])]
    sid_list_all2=[sid_left_list(frame_list, df) for frame_list in groups]
    sid_list_2look=[k for k,g in enumerate(sid_list_all2) if len(set(g)) != 1]
    frames=[f_list[0] for k,f_list in enumerate(groups) if k not in sid_list_2look]
    if len(sid_list_2look) !=0:
        for i in sid_list_2look:
            inds=[sid_list_all2[i].index(z) for z in set(sid_list_all2[i])]
            frames+=[groups[i][ind] for ind in inds]
        frames=sorted(frames)
    return frames

def get_itc_count(n,na_itc):
    return np.asarray([1 if i== n else 0 for i in na_itc])

def itc_duration(df,tag,tags):
    mice_list=[ta for ta in tags if ta!=tag]
    mice_list.append('UK')
    data=[]
    durations=df.Time.diff()
    ita_status=[mice[tag] for mice in df.Interactions]
    for ta in mice_list:
        values= np.array([1 if ta in i else 0 for i in ita_status])
        values=values[:-1]*durations[1:]
        values=np.append(values,np.nan)
        data.append(values)
    alone=[]
    for i in range(len(df.frame.values)):
        if i != len(df.frame.values):
            itc=[data[z][i]==0 for z in range(len(data))]
        if np.all(itc):
            alone.append(1)
        else:
            alone.append(0)
    alone=np.asarray(alone)[:-1]*durations[1:]
    alone=np.append(alone,np.nan)
    df_dic={i:z for i,z in zip(mice_list,data)}
    df_itc=pd.DataFrame(df_dic)
    df_itc=df_itc.fillna(value=0)
    vals=np.asarray([df_itc[col].values for col in df_itc.columns])
    na_itc=np.apply_along_axis(np.count_nonzero,1,vals.T)
    n_animal_cols={f'itc_{count+1}': get_itc_count(count+1,na_itc) 
                   for count in range(len(tags)-1)}
    for k,v in n_animal_cols.items():
        durs=v[:-1]*durations[1:]
        durs=np.append(durs,np.nan)
        df_itc[k]=durs
    df_itc['alone']=alone
    df_itc['frame']=df.frame.values
    return df_itc






