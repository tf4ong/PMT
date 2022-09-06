import pandas as pd 
from prt_utils import mouse_matching as mm
import os
import numpy as np
import prt_utils.trajectory_analysis_utils as ta
import prt_utils.configloader as config_loader
import sys
from prt_utils.track_utils import *
import itertools
from prt_utils.generate_vid import generate_RFID_video,create_validation_Video,generate_video
from prt_utils.detect_utils import yolov4_detect_vid
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
"""
Open arena (i.e. no entrance/exit arena) speed not optomized
may need adiitionl methods for faster inference if long videos
faster framerate mayhelp, only tested upto 40fps
"""
class PRT_analysis:
    def __init__(self,path,config_path,n_mice=None,rfid_tags=None):
        self.path=path 
        self.config_path=config_path
        self.config_dic_analysis=config_loader.analysis_config_loader(config_path)
        self.config_dic_detect=config_loader.detect_config_loader(config_path)
        self.config_dic_tracking=config_loader.tracking_config_loader(config_path)
        self.config_dic_dlc=config_loader.dlc_config_loader(config_path)
        if os.path.exists(self.path+'/RFID_tracks.csv'):
            print('Already analyzed. Found RFID_tracks.csv file in path')
            self.load_data()
            #reload=input('Load analyzed data? Y/N')
            #if reload.lower() =='y':
            #    self.load_data()
            #else:
            #    pass
        if rfid_tags is not None and n_mice is not None:
            if not os.path.exists(self.path+'/logs.txt'):
                tags = [str(i) for i in rfid_tags]
                if len(tags)>1:
                    tags = ','.join(tags)
                else:
                    tags = tags[0]
                with open(self.path+'/logs.txt','w') as file:
                    file.write(f'mice:{n_mice}\n')
                     #for tag in rfid_tags:
                    file.write(f'Tags: {tags}')
        with open(path+'/'+'logs.txt','r') as f:
            tags=f.readlines()
            self.tags=[int(i) for i in tags[1][6:].split(',')]
        return
    def load_data(self):
        convert_dict={'sort_tracks':eval,'RFID_tracks':eval,'ious_interaction':eval,
                      'Interactions':eval,'motion_roi':eval,'RFID_matched':eval,
                      'Activity':eval}
        try:
            self.df_tracks_out=pd.read_csv(self.path+'/RFID_tracks.csv',converters=convert_dict)
            print(f'Loaded RFID_tracks.csv at path {self.path}')
        except Exception as e:
            print(e)
            print('confimr that mice detections are in folder')
        return
    def detect_mice(self,write_vid=False):
        yolov4_detect_vid(self.path,self.config_dic_detect,write_vid)
        return
    def load_RFID(self):
        #try:
        self.df_RFID=mm.RFID_readout(self.path,self.config_dic_analysis,int(len(self.tags)))
        #except Exception as e:
        #    print(e)
        #    print('Confirm correct folder path')
            #sys.exit(0)
        if len(self.tags) !=1:
            n_RFID_readings=len(self.df_RFID[self.df_RFID['RFID_readings'].notnull()])
        else:
            n_RFID_readings=0
        duration=self.df_RFID.iloc[-1]['Time']-self.df_RFID.iloc[0]['Time']
        print(f'{n_RFID_readings} Tags were read in {duration} seconds')
    def load_dets(self):
        self.df_tracks=mm.read_yolotracks(self.path,self.config_dic_analysis,self.config_dic_tracking,
                                          self.df_RFID,len(self.tags))
        if self.config_dic_analysis['entrance_reader'] is None:
            #self.df_tracks=mm.reconnect_tracks_ofa(self.df_tracks,len(self.tags))
            pass
        else:
            reconnect_ids=mm.get_reconnects(self.df_tracks)
            self.df_tracks,id2remove=mm.spontanuous_bb_remover(reconnect_ids,self.df_tracks,
                                                               self.config_dic_analysis,self.config_dic_tracking)       
            reconnect_ids=mm.reconnect_id_update(reconnect_ids,id2remove)
            trac_dict_leap=mm.reconnect_leap(reconnect_ids,self.df_tracks,self.config_dic_tracking['leap_distance'])
            self.df_tracks=mm.replace_track_leap_df(trac_dict_leap,self.df_tracks,self.config_dic_analysis)
        self.df_tracks=mm.Combine_RFIDreads(self.df_tracks,self.df_RFID)
        return self.df_tracks
    def RFID_match(self,report_coverage=True,save_csv=True):
        if len(self.tags)!=1:
            self.df_tracks_out,self.validation_frames=mm.RFID_matching(self.df_tracks,self.tags,self.config_dic_analysis,
                                                                       self.path)
            self.df_tracks_out=mm.match_left_over_tag(self.df_tracks_out,self.tags,self.config_dic_analysis)
            self.df_tracks_out=mm.tag_left_recover_simp(self.df_tracks_out,self.tags)
            self.df_tracks_out=mm.interaction2dic(self.df_tracks_out,self.tags,self.config_dic_analysis['itc_slack'])
            self.df_tracks_out=self.df_tracks_out[['frame','Time','sort_tracks','RFID_tracks','ious_interaction','Interactions',
                                                   'motion','motion_roi','RFID_matched','RFID_readings','Correction',
                                                   'Matching_details']]
            if save_csv:
                self.df_tracks_out.to_csv(self.path+'/RFID_tracks.csv')
                #print(f'csv file saved at {self.path+"/RFID_tracks.csv"}')
            if report_coverage:
                coverage=mm.coverage(self.df_tracks_out)
            return self.df_tracks_out,self.validation_frames,coverage
        else:
            self.validation_frames=[]
            self.df_tracks['lost_tracks']=self.df_tracks['sort_tracks'].values
            self.df_tracks_out=mm.tag_left_recover_simp(self.df_tracks,self.tags)
            self.df_tracks_out.to_csv(self.path+'/RFID_tracks.csv')
            if report_coverage:
                coverage=mm.coverage(self.df_tracks_out)
            self.df_tracks_out=mm.interaction2dic(self.df_tracks_out,self.tags,self.config_dic_analysis['itc_slack'])
            self.df_tracks_out=self.df_tracks_out[['frame','Time','sort_tracks','RFID_tracks','ious_interaction','Interactions',
                                                   'motion','motion_roi','RFID_matched']]
            self.df_tracks_out['Correction']=[[] for i in range(len(self.df_tracks_out))]
            self.df_tracks_out['Matching_details']=[[] for i in range(len(self.df_tracks_out))]
            self.df_tracks_out['RFID_readings']=[[] for i in range(len(self.df_tracks_out))]
            if save_csv:
                self.df_tracks_out.to_csv(self.path+'/RFID_tracks.csv')
                print(f'csv file saved at {self.path+"/RFID_tracks.csv"}')
            return self.df_tracks_out,self.validation_frames,coverage
    def find_activte_mice(self,save_csv=True):
        self.df_tracks_out['Activity']=[mm.get_tracked_activity(motion_status,motion_roi,RFID_tracks,self.tags) for motion_status,
                                        motion_roi,RFID_tracks in zip(self.df_tracks_out['motion'].values,
                                                                      self.df_tracks_out['motion_roi'].values,
                                                                      self.df_tracks_out['RFID_tracks'].values)]
        if save_csv:
            self.df_tracks_out.to_csv(self.path+'/RFID_tracks.csv')
            #print(f'csv file saved at {self.path+"/RFID_tracks.csv"}')
        return self.df_tracks_out
    def load_dlc_bpts(self):
        print('Loading Deeplabcut body parts to PRT')
        columns=['frame']+ self.config_dic_dlc['dbpt']
        dics={y: eval for y in columns}
        df_bpts=pd.read_csv(self.path+'/'+'dlc_bpts.csv',converters=dics)
        df_dbpt_columns=[f'df_bpts["{i}"]' for i in self.config_dic_dlc['dbpt']]
        df_bpts['bpts']=eval('+'.join(df_dbpt_columns))
        df_bpts['frame']=range(len(df_bpts))
        df_bpts=df_bpts.drop(columns=self.config_dic_dlc['dbpt'])
        columns=['bboxes']
        #self.df_tracks_out.to_csv('test.csv')
        self.df_tracks_out=self.df_tracks_out[['frame', 'Time','sort_tracks', 'RFID_tracks', 
                                               'ious_interaction', 'Interactions','motion', 
                                               'motion_roi', 'RFID_matched', 'Activity']]
        self.df_tracks_out=pd.merge( self.df_tracks_out,df_bpts, on='frame')
        dbpts=[mm.rfid2bpts(bpts,RFIDs,self.config_dic_dlc['dbpt_box_slack'],bpt2look=self.config_dic_dlc['dbpt_distance_compute']) 
               for bpts,RFIDs in zip(self.df_tracks_out['bpts'].values,self.df_tracks_out['RFID_tracks'].values)]
        self.df_tracks_out['dbpt2look']=[i[0] for i in dbpts]
        self.df_tracks_out['undetemined_bpt']=[i[1] for i in dbpts]
        list_bpts=list(map(sublist_decompose,self.df_tracks_out.dbpt2look.values.tolist()))
        for i in self.tags: 
            exec(f'list_bpt_{str(i)}=[]')
            for y in list_bpts:
                bpts=[v for v in y if v[3]==i]
                exec(f'list_bpt_{str(i)}.append(bpts)')
            self.df_tracks_out[f'{i}_bpts']=eval(f'list_bpt_{str(i)}')
        rows=self.df_tracks_out.apply(lambda x:mm.bpt_distance_compute(x,self.tags,self.config_dic_dlc['dbpt_int']),axis=1)
        new_cols=[str(list(i)[0]) + '_'+str(list(i)[1]) for i in itertools.combinations(self.tags, 2)]
        for name,idx in zip(new_cols,range(len(new_cols))):
            self.df_tracks_out[name]=[dists[idx] for dists in rows]
        print('Finished Loading Deeplabcut body parts to PRT')
        self.df_tracks_out.to_csv(self.path+'/RFID_tracks.csv')
        return 

    def compile_travel_trjectories(self,dlc=False):
        msg='Generating Individual Rodent Trajectory'
        pbar=tqdm(total=len(self.tags),position=0,leave=True,desc=msg)
        if not os.path.exists(self.path+'/trajectories'):
            os.mkdir(self.path+'/trajectories')
        for tag in self.tags:
            list_df=ta.location_compiler(tag,self.df_tracks_out,dlc,lim=5)
            if not os.path.exists(self.path+'/trajectories'+f'/{tag}'):
                os.mkdir(self.path+'/trajectories'+f'/{tag}')
            count=0
            if list_df != []:
                if dlc: 
                    list_df=list(map(ta.dbpts2xy,[[self.config_dic_dlc['dbpt'],df] for df in list_df]))
                for tracks in list_df:
                    tracks.to_csv(self.path+'/trajectories'+f'/{tag}'+f'/track_{count}.csv')
                    count+=1
                df_t=self.df_tracks_out[['Time','frame']]
                df_tag=pd.concat(list_df)
                df_tag=df_tag.sort_values(by=['frame'])
                df_tag_c=pd.merge(df_t,df_tag,on='frame',how='outer')
                df_itc=mm.itc_duration(self.df_tracks_out,tag,self.tags)
                df_tag_c=pd.merge(df_tag_c,df_itc,on='frame',how='outer')
                df_tag_c.iloc[np.where(df_tag_c['x'].isnull())[0],4:]=np.nan
                #df_tag_c.loc[df_tag_c.isnull().any(axis=1), :] = np.nan
                df_tag_c.to_csv(self.path+'/'+f'{tag}.csv')
            pbar.update(1)
        return 
    
    def generate_labeled_video(self,dlc_bpts=False,plot_motion=False,out_folder=None):
        generate_RFID_video(self.path,self.df_RFID,self.tags,self.df_tracks_out,\
                               self.validation_frames,self.config_dic_analysis,self.config_dic_dlc,plot_motion,out_folder=out_folder)
        
    def generate_validation_video(self,out_folder='None'):
        create_validation_Video(self.path,self.df_tracks_out,self.tags,self.config_dic_analysis,output=None)
    def generate_track_video(self,out_folder='None'):
        generate_video(self.path,self.df_tracks_out,self.tags,self.config_dic_analysis,output=None)

import shutil
from prt_utils.extract_frames2label import extract_frames
import scripts.voc_convert as vc
import prt_utils.train_utils as train_utils
from random import shuffle
from mAP import eval_tools as et
from prt_utils import detect_utils as du
#still mising conversion of pvoc file in the labelimg pathway


class PRT_train:
    def __init__(self,path2config,creat_new=False):
        self.main_folder = path2config
        if creat_new:
            if not os.path.exists(path2config):
                os.mkdir(path2config)
            shutil.copyfile('./configs/config.ini',f'{path2config}/config.ini')
            print('finished creating config.ini file at {path2config}')
            print('please edit config files to desired settings and load with load_config()')
            self.config_path=f'{path2config}/config.ini'
            config_loader.set_value(self.config_path,'PRT_Train','outpath',f'{path2config}/trainning/')
            if not os.path.exists(f'{path2config}/trainning/'):
                os.mkdir(f'{path2config}/trainning/')
            self.train = f'{path2config}/trainning/'
        else:
            self.config_path=f'{path2config}/config.ini'
            self.load_config()
            print(f'Loaded config path at {path2config}/config.ini')
            if not os.path.exists(f'{path2config}/trainning/'):
                os.mkdir(f'{path2config}/trainning/')
            self.train = f'{path2config}/trainning/'
        self.vid_folder = f'{path2config}/label_imgs'
        self.vid_paths =[]
    def load_config(self):
        self.config_dic_train = config_loader.train_config_loader(self.config_path)
        if not os.path.exists(f'{self.main_folder}/names.txt'):
            with open(f'{self.main_folder}/names.txt','w') as out:
                for v in self.config_dic_train['classes'].values():
                    out.write(f'{v}\n')
                out.close()
        return
    def add_vids(self,vid_paths):
        self.vid_paths = vid_paths
    def extractframes(self):
        if not os.path.exists(self.vid_folder):
            os.mkdir(self.vid_folder)
        if len(self.vid_paths) ==0:
            print('please add videos to folder')
            return
        for vid in self.vid_paths:
            extract_frames(vid,self.config_dic_train['frames2pic'] ,self.vid_folder)
        print('Finished extracting frames from videos')
        print('label frames with labelimg using label_imgs() or use another too1649974680_img4l which generates Pascal VOC format')
    def label_imgs(self):
        temp_vid = self.vid_folder
        temp_main = self.main_folder
        if os.listdir(self.vid_folder) ==0:
            print('extract images to label by running extractframes')
            return
        if ' ' in self.vid_folder or '' in self.main_folder: 
            space_pos = [i for i,v in enumerate(self.vid_folder) if v == ' ']
            space_pos2 = [i for i,v in enumerate(self.main_folder) if v == ' ']
            l1=[s for s in self.vid_folder]
            l2=[s for s in self.main_folder]
            spacer=0
            spacer2=0
            for i in space_pos:
                l1.insert(i+spacer, '\\')
                spacer+=1
            for i in space_pos2:
                l2.insert(i+spacer2, '\\')
                spacer2+=1
            temp_vid = ''.join(l1)
            temp_main = ''.join(l2)
        os.system(f'labelImg {temp_vid} {temp_main}/names.txt')
        return

    def evaluate_weights(self,plot_predict=True):
        self.evaluate_path = self.main_folder+'/evaulation_results'
        self.config_dic_detect=config_loader.detect_config_loader(self.config_path)
        if not os.path.exists(self.evaluate_path):
            os.mkdir(self.evaluate_path)
        weight_name = os.path.basename(self.config_dic_detect['weightpath'][0]) 
        results_path = self.evaluate_path+'/'+ weight_name
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        ground_truth_dic = et.read_gt(self.vid_folder)
        if len(ground_truth_dic) == 0:
            raise Exception('No ground truth labels found, check if there are label images')
        img_list = [self.vid_folder +'/'+ i for i in os.listdir(self.vid_folder) if i[-4:]=='.png' or i[-4:]=='.jpg']
        predicts_dic = du.yolov4_detect_images(img_list,self.config_dic_detect,results_path,save_out=True)
        total_fn,total_fp,mean_ap = et.get_mAP(ground_truth_dic,predicts_dic)
        print('#####################################################\n')
        print(f"Results for weights at path: {self.config_dic_detect['weightpath'][0]}\n")
        print(f"Score Threshold: {self.config_dic_detect['score']}, IOU Threshold: {self.config_dic_detect['iou']}\n")
        print('Total false positive in this image set:',total_fp,'\n',
            'Total false negative in this image set',total_fn,'\n',
            'Mean average precision (mAP):',mean_ap)
        if not os.path.exists(self.evaluate_path+'/results.csv'):
            with open(self.evaluate_path+'/results.csv','w') as file:
                file.write('weight,weight_path,score,iou,fn,fp,mAP\n')
        with open(self.evaluate_path+'/results.csv','a') as file:
            file.write(f"{weight_name},{self.config_dic_detect['weightpath'][0]},{self.config_dic_detect['score']},{self.config_dic_detect['iou']},{total_fp},{total_fn},{mean_ap}")
        if plot_predict:
            et.plt_results(predicts_dic,ground_truth_dic,results_path)
        print('Test another weight and adjust score/iou threshold as necessary')
        return 

    def pvocTrainImport(self,pvoc_path):
        #coverts pvoc labels to train labels
        # discarded due to train function not complete in tf2-yolov4
        ims = vc.get_labled_imgs(pvoc_path)
        shuffle(ims)
        train=int(len(ims)*self.config_dic_train['trainfraction']) 
        train_ims = ims[:train]
        test_ims = ims[train:]
        print('Generating Trainning set')
        self.train_txt=vc.parse_text(train_ims,pvoc_path,self.train,'train',use_difficult_bbox= True)
        print('Generating Test set')
        self.eval_txt=vc.parse_text(test_ims, pvoc_path,self.train,'test',use_difficult_bbox= True)
        config_loader.set_value(self.config_path,'PRT_Train','trainpath_txt',self.train_txt)
        config_loader.set_value(self.config_path,'PRT_Train','testpath_text',self.eval_txt)
        print('Ready to train, export folder to darknet for trainning, good luck!')
        return

    def generate_trainset(self):
        if not os.path.exists(f'{self.train}yolov4-obj.cfg'):
            shutil.copyfile('./configs/yolov4-obj.cfg',f'{self.train}yolov4-obj.cfg')
        if not os.path.exists(self.vid_folder):
            print('No images labeld, extract and label images using extractframes and label_imgs')
            return
        else:
            if os.path.exists(f'{self.train}train_set') or os.path.exists(f'{self.train}train.txt'):
                print('Trainset exists, please vertified and delete the train_set and train.txt files if necessary')
            elif os.path.exists(f'{self.train}test_set') or os.path.exists(f'{self.train}test.txt'):
                print('Testset exists, please vertified and delete the test_set and test.txt files if necessary')
            else:
                train_utils.generate_sets(self.vid_folder,self.config_dic_train['trainfraction'],self.train)
        if not os.path.exists(f'{self.train}obj.names'):
            with open(f'{self.train}obj.names','w') as out:
                for v in self.config_dic_train['classes'].values():
                    out.write(f'{v}\n')
                out.close()
        train_utils.generate_pathtext(self.train,len(self.config_dic_train['classes']))
        train_utils.load_train_settings(self.config_dic_train,self.train)
        print('Please transfer folder to train in darknet, in the colab notebook or docker provided')
        return


