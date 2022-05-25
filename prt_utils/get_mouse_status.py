import pandas as pd 
import tqdm
import os
from datetime import datetime,timedelta
import numpy as np
import math
import traja
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import itertools


def status_update(tag,track_ids,tracked_tags,nmice):
    '''
    monitors if the mice is in the cage
    '''
    if tag in tracked_tags:
        return 1
    if len(track_ids) ==nmice:
        return 1
    else:
        return 0

def interaction_correcter(tags,iou_dic,nmice):
    '''
    gets the specific mouse the tagged mouse is interacting with
    '''
    template=['UK' for count in range(nmice-1)]
    if template not in iou_dic.values():

        return iou_dic
    else:
       checker={sid:sids for sid,sids in iou_dic.items()}
       keys=list(checker.keys())
       print('1')
       if checker[keys[0]]==[keys[1]]:
           dic_return={sid:[] for sid in tags}
           for tag in tags:
               dic_return[tag]=[ta for ta in tags if ta != tag]
           return dic_return
       else:
           return iou_dic

def read_mouse_interactions(folder,path,tags,corner_thres,corner_cent):
        c_tag=[str(list(i)[0]) + '_'+str(list(i)[1]) for i in itertools.permutations(tags, 2)]
        pathin=path+folder+'/mouse_status.csv'
        columns=['Time','Tracked','Interaction_tracks','Activity']+[str(tag) for tag in tags]+[str(tag)+'_status' for tag in tags]+c_tag
        dics={i: eval for i in columns}
        df1=pd.read_csv(pathin,converters=dics)
        df1['video']=folder
        df1['Duration']=df1['Time'].diff()
        for tag in tags:
            df1[str(tag)+'in_corner']=[corner_check(tag,tracked_tags,track,corner_thres,corner_cent) for tracked_tags,track in zip(df1.Tracked.values,df1.Interaction_tracks.values)]
        for tag in tags:
            df1[str(tag)+'out_corner']=[out_ccheck(tag,tracked_tags,track,corner_thres,corner_cent) for tracked_tags,track in zip(df1.Tracked.values,df1.Interaction_tracks.values)]
        for tag in tags:
            values=df1[str(tag)+'in_corner'].values[:-1]*df1['Duration'].values[1:]
            df1[str(tag)+'corner_dur'] =np.insert(values,len(values),np.nan)
            df1=df1.drop(columns=[str(tag)+'in_corner'])
        for tag in tags:
            values=df1[str(tag)+'out_corner'].values[:-1]*df1['Duration'].values[1:]
            df1[str(tag)+'ocorner_dur'] =np.insert(values,len(values),np.nan)
            df1=df1.drop(columns=[str(tag)+'out_corner'])
        for tag in tags:
            values=df1[str(tag)+'_status'].values[:-1]*df1['Duration'].values[1:]
            df1[str(tag)+'in_cage_dur'] =np.insert(values,len(values),np.nan)
            df1=df1.drop(columns=[str(tag)+'_status'])
        df1=df1.drop(columns=['Interaction_tracks'])
        return df1

def read_RFID_data(pathin,tags,n_mice):
    c_tag=[str(list(i)[0]) + '_'+str(list(i)[1]) for i in itertools.permutations(tags, 2)]
    columns=['Time','track_id','Tracked','UnTracked','Interactions','Activity']+c_tag
    dics={i: eval for i in columns}
    df1=pd.read_csv(pathin,converters=dics)
    df1['Interactions']=[interaction_correcter(tags,iou_dic,n_mice) for iou_dic in df1.Interactions.values]
    df1['N_Cage']=[len(sids) for sids in df1.track_id.values]
    for i in tags:
        df1[str(i)+'_status']=[status_update(i,track_ids,tracked_tags,n_mice) for track_ids,tracked_tags in zip(df1.track_id.values,df1.Tracked.values)]
    for i in tags:
        df1[str(i)]=[values[i] for values in df1.Interactions.values]
    """
    for i in tags:
        df1[str(i)+'n_mice']=[len(values) for values in df1[str(i)].values]
    for i in tags:
        for n in range(n_mice):
            df1[str(i)+f'_{n}_mice']=[1 if mice ==n else 0 for mice in df1[str(i)+'n_mice'].values]
    """
    c_drop=['Unnamed: 0', 'bboxes', 'sort_tracks', 'track_id', 'ious',
           'Entrance', 'Cage', 'sort_entrance_dets', 'Entrance_ids', 'Cage_ids',
           'sort_cage_dets', 'level_0', 'RFID_readings','RFID_tracks',
           'RFID_matched', 'lost_tracks', 'track_cage','Tracked_marked',
           'RFID_tracks_cage', 'UnTracked',
           'ious_interaction', 'Interactions', 'N_Cage']
    df1=df1.drop(columns=c_drop) 
    return df1

def bbox_to_centroid(bbox):
    '''
    returns the centroid of the bbox
    '''
    cX=int((bbox[0]+bbox[2])/2)
    cY=int((bbox[1]+bbox[3])/2)
    return [cX,cY]


def Distance(centroid1,centroid2):
    ''' 
    calculates the centronoid distances between bbs
    intake centronoid
    '''
    dist = math.sqrt((centroid2[0] - centroid1[0])**2 + (centroid2[1] - centroid1[1])**2)
    return dist

def in_corner(track,corner_thres,corner_cent):
    dist=[Distance(bbox_to_centroid(track),i) for i in corner_cent]
    if min(dist)<=corner_thres:
        return 1
    else:
        return 0

def out_corner(track,corner_thres,corner_cent):
    dist=[Distance(bbox_to_centroid(track),i) for i in corner_cent]
    if min(dist)<=corner_thres:
        return 0
    else:
        return 1

def corner_check(tag,tracked_tags,track,corner_thres,corner_cent):
    if tag not in tracked_tags:
        return 0
    else: 
        track2check=[t for t in track if t[4]==tag][0]
        return in_corner(track2check,corner_thres,corner_cent)
    
def out_ccheck(tag,tracked_tags,track,corner_thres,corner_cent):
    if tag not in tracked_tags:
        return 0
    else:
        track2check=[t for t in track if t[4]==tag][0]
        return out_corner(track2check,corner_thres,corner_cent)


def read_midfs(path):
    df=pd.read_csv(path)
    df['Time']=[datetime.utcfromtimestamp(t) for t in df['Time']]
    tag=os.path.basename(path).split('_')[0]
    return df,tag


class mouse_status:
    def __init__(self,path,corners,tags,n_mice,corner_thres=50):
        self.path=path
        self.corner_cent= [bbox_to_centroid(i) for i in corners]
        self.tags=tags
        self.n_mice=n_mice
        self.corner_thres=corner_thres
    def read_RFID_tracks(self):
        folders=[folder for folder in os.listdir(self.path) if folder[-4:]!='.csv']
        pbar=tqdm.tqdm(total=len(folders),position=0, leave=True)
        for folder in folders:
            pathin=self.path+folder+'/RFID_tracks.csv'
            df1=read_RFID_data(pathin,self.tags,self.n_mice)
            file_name= os.path.dirname(pathin)+'/mouse_status.csv'
            df1.to_csv(file_name)
            pbar.update(1)
        del df1
    def assemble_mouse_status(self):
        print('Loading Mouse Status data files')
        folders=[folder for folder in os.listdir(self.path) if folder[-4:]!='.csv']
        pbar=tqdm.tqdm(total=len(folders),position=0, leave=True)
        df_list=[]
        for folder in folders:
            print(f' Reading {folder}')
            df_list.append(read_mouse_interactions(folder,self.path,self.tags,self.corner_thres,self.corner_cent))
            pbar.update(1)
        print('Loaded all related files')
        self.df_final=pd.concat(df_list)
        del df_list
        self.df_final=self.df_final.sort_values(by=['Time'])
        self.df_final=self.df_final.drop(columns=['Unnamed: 0'])
        self.df_final.to_csv(self.path+'status.csv')
       #return self.df_final
    def output_mice_interactions(self):
        self.mouse_interactions={}
        for tag in self.tags:
            mice_list=[ta for ta in self.tags if ta!=tag]
            mice_list.append('UK')
            data=[]
            for ta in mice_list:
                values= np.array([1 if ta in i else 0 for i in self.df_final[str(tag)].values])
                data.append(values)
            list_nmice={}
            for i in range(len(data)):
                exec(f'n_{str(i)}_mouse=[]')
                exec(f'list_nmice[i]=n_{str(i)}_mouse')
            for i in sum(data):
                for z in list_nmice.keys():
                    if z != i:
                        list_nmice[z].append(0)
                    else:
                        list_nmice[z].append(1)
            v2check=list_nmice[0]
            for i in list_nmice.keys():
               list_nmice[i]=(np.asarray(list_nmice[i][:-1])*np.asarray(self.df_final['Duration'].values[1:])).tolist()
               list_nmice[i]=np.insert(list_nmice[i],len(list_nmice[i]),np.nan)
            mice_list=[str(i) for i in mice_list]
            data=[value[:-1]*self.df_final['Duration'].values[1:] for value in data]
            data=[np.insert(value,len(value),np.nan) for value in data ]
            mice_list.append('Time')
            data.append(self.df_final['Time'].to_list())
            dic={mice_list[i]:data[i] for i in range(len(mice_list))}
            dic.update(list_nmice)
            df=pd.DataFrame.from_dict(dic)
            df[str(tag)+'ocorner_dur']=self.df_final[str(tag)+'ocorner_dur'].to_list()
            df[str(tag)+'corner_dur']=self.df_final[str(tag)+'corner_dur'].to_list()
            df[str(tag)+'in_cage_dur']=self.df_final[str(tag)+'in_cage_dur'].to_list()
            df=df.drop(columns=[0])
            alone_marker=[]
            track2check=self.df_final['Tracked'].values
            for i in range(len(v2check)):
                if v2check[i] ==0:
                    alone_marker.append(0)
                else:
                    if tag in track2check[i]:
                        alone_marker.append(1)
                    else:
                        alone_marker.append(0)
            alone_marker=(np.asarray(alone_marker[:-1])*np.asarray(self.df_final['Duration'].values[1:])).tolist()
            alone_marker=np.insert(alone_marker,len(alone_marker),np.nan)
            df[0]=alone_marker
            df.to_csv(f'{self.path}{tag}_interactions.csv')
            self.mouse_interactions[tag]=df
            del mice_list
            del data
            del dic
            del list_nmice
        del df
    def trajectory_analysis(self):
        pbar=tqdm.tqdm(total=len(self.tags),position=0, leave=True)
        folders=[self.path+folder+'/data/' for folder in os.listdir(self.path) if folder[-4:]!='.csv' and folder[-4:]!='.txt']
        columns=['Centroid']
        dics={i: eval for i in columns}
        self.df_kinetics={}
        for tag in self.tags:
            print(f' Extracting travel trajectory for mouse {tag}')
            df_list=[]
            folder_sub=[folder1+f'{str(tag)}_dfs' for folder1 in folders]
            pbar2=tqdm.tqdm(total=len(folder_sub),position=0, leave=True)
            for f_sub in folder_sub:
                for track in os.listdir(f_sub):
                    df=pd.read_csv(f_sub+'/'+track,converters=dics)
                    df['x']=[i[0] for i in df.Centroid.values]                                                                                                                                                                                                                                  
                    df['y']=[i[1] for i in df.Centroid.values]
                    t_start=df.iloc[0]['Timestamp']
                    df['Time']=[i-t_start for i  in df['Timestamp'].values]
                    df['Turn_Angle']=df.traja.calc_turn_angle()
                    df_list.append(df)
                pbar2.update(1)
            df_final=pd.concat(df_list)
            df_final=df_final.sort_values(by=['Time'])
            df_final=df_final.drop(columns=['Unnamed: 0', 'delta_time', 'Centroid_X', 'Centroid_Y',
               'x1', 'y1', 'x2', 'y2', 'Centroid','bbox', 'x', 'y', 'Time'])
            df_final['Mouse']=tag
            df_final['Turn_Angle']=[abs(angle) for angle in df_final['Turn_Angle'].values]
            df_final['N_Turns']=[0 if i ==0 or np.isnan(i)  else 1  for i in df_final.Turn_Angle.values]
            df_final.to_csv(f'{self.path}{str(tag)}_kinetics.csv')
            self.df_kinetics[f'{str(tag)}']=df_final
            pbar.update(1)
        print('All {len(self.tags)} mice travel trajector analyzed and stored')
        return self.df_kinetics
    def aggregate_ms(self, time_bin='1H'):
        for i in self.mouse_interactions.keys():
            self.mouse_interactions[i]['Time'] =[datetime.utcfromtimestamp(t) for t in self.mouse_interactions[i]['Time']] 
            self.mouse_interactions[i]=self.mouse_interactions[i].set_index('Time')
            self.mouse_interactions[i]=self.mouse_interactions[i].groupby(pd.Grouper(freq=time_bin)).sum()
            self.mouse_interactions[i]=self.mouse_interactions[i].reset_index()
            renames={str(i)+'ocorner_dur':'ocorner_dur',str(i)+'corner_dur':'corner_dur',str(i)+'in_cage_dur':'in_cage_dur'}
            self.mouse_interactions[i]=self.mouse_interactions[i].rename(columns=renames)
            c_drop=[str(tag) for tag in self.mouse_interactions.keys() if tag!=i]
            self.mouse_interactions[i]=self.mouse_interactions[i].drop(columns=c_drop)
            self.mouse_interactions[i]['Mouse']=i
            self.mouse_interactions[i]=self.mouse_interactions[i].drop(columns=['UK'])
        self.mouse_interactions=[i for i in self.mouse_interactions.values()]
        self.mouse_interactions=pd.concat(self.mouse_interactions)
        self.mouse_interactions.to_csv(f'{self.path}Status_Summary.csv',index=False)
    def aggregate_tt(self, time_bin='1H'):
        self.df_ks=[]
        for i in self.df_kinetics.keys():
            df_final= self.df_kinetics[i]
            df_final['Timestamp']=[datetime.utcfromtimestamp(t) for t in df_final['Timestamp']] 
            df_final=df_final.set_index('Timestamp')
            total_distance=df_final['Dist'].groupby(pd.Grouper(freq=time_bin)).sum().rename('Total Distance')
            average_speed=df_final['Speed'].groupby(pd.Grouper(freq=time_bin)).mean().rename('Average_Speed')
            max_speed=df_final['Speed'].groupby(pd.Grouper(freq=time_bin)).max().rename('Max_Speed')
            anagle_variation=df_final['Turn_Angle'].groupby(pd.Grouper(freq=time_bin)).std().rename('Angle_Variation')
            anagle_mean=df_final['Turn_Angle'].groupby(pd.Grouper(freq=time_bin)).mean().rename('Mean_Turn_Angle')
            mean_turns=df_final['N_Turns'].groupby(pd.Grouper(freq=time_bin)).mean().rename('Mean_Turns')
            total_turns=df_final['N_Turns'].groupby(pd.Grouper(freq=time_bin)).sum().rename('Total_Turns')
            df_final=pd.concat([total_distance,average_speed,max_speed,anagle_variation,anagle_mean,mean_turns,total_turns],axis=1)
            df_final['Mouse']=i
            df_final=df_final.fillna(0)
            df_final=df_final.reset_index()
            self.df_ks.append(df_final)
        self.df_ks=pd.concat(self.df_ks)
        self.df_ks.to_csv(f'{self.path}_kinetics_summary.csv')
        return self.df_ks
            
        
        

















    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


