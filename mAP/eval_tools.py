import numpy as np
import os
from tqdm import tqdm 
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

def plt_results(pred_dic,ground_truth_dic,path):
    print('Writing prediction to images')
    pbar = tqdm(total=len(pred_dic),leave=True,position=0)
    if not os.path.exists(path+'/labels'):
        os.mkdir(path+'/labels')
    for i,v in pred_dic.items():
        bname = os.path.basename(i)
        img = cv2.imread(i)
        gts = ground_truth_dic[i[:-4]+'.txt']
        for pred in v:
            cv2.rectangle(img, (pred[0], pred[1]), (pred[2], pred[3]), (0,0,255), 3)
        for gt in gts:
            cv2.rectangle(img, (gt[0], gt[1]), (gt[2], gt[3]), (0,255,0), 3)
        cv2.imwrite(path+'/labels'+f'/{bname}',img)
        pbar.update(1)
    print('Results can be viewed at {path}, green -> ground Truth; red -> predictions')
    return 


####code below done by Hao Hu 
def gt_predicts_numpy(gt,predicts):
    # translate the dics into some ndarray inside a list
    # each element in the list is one frame
    pred_bboxes, pred_labels,pred_scores, gt_bboxes,gt_labels,gt_difficult = [],[],[],[],[],[]
    for idx, pred_each_frame in predicts.items():
        gt_each_frame = gt[idx[:-4]+'.txt']
        batch_size1 =  len(pred_each_frame)
        batch_size2 =  len(gt_each_frame)
        #assert(batch_size1 == batch_size2)
        pred_bboxes_each_f  = np.zeros((batch_size1,4))
        pred_labels_each_f  = np.zeros((batch_size1,1))
        pred_scores_each_f  = np.zeros((batch_size1,1))
        gt_bboxes_each_f    = np.zeros((batch_size2,4))
        gt_labels_each_f    = np.zeros((batch_size2,1))
        gt_difficult_each_f = np.zeros((batch_size2,1))
        for i in range(batch_size1):
            pred_bboxes_each_f[i,:]  = np.array(pred_each_frame[i][:4])
            pred_bboxes_each_f[i,2:]+= 1 
            pred_labels_each_f[i,:]  = np.array([0])
            pred_scores_each_f[i,:]  = np.array(pred_each_frame[i][-2])
        for j in range(batch_size2):
            gt_bboxes_each_f[j,:]    = np.array(gt_each_frame[j][:4])
            gt_bboxes_each_f[j,2:]  += 1
            gt_labels_each_f[j,:]    = np.array(gt_each_frame[j][-1]) 
            
        
        order = pred_scores_each_f.reshape(-1).argsort()[::-1]
        
        
        pred_bboxes.append(pred_bboxes_each_f[order])
        pred_labels.append(pred_labels_each_f[order])
        pred_scores.append(pred_scores_each_f[order])
        gt_bboxes.append(gt_bboxes_each_f)
        gt_labels.append(gt_labels_each_f)
        gt_difficult.append(gt_difficult_each_f)
        
        
    return pred_bboxes, pred_labels,pred_scores, gt_bboxes,gt_labels,gt_difficult

def bbox_iou(bbox_a, bbox_b, offset=0):
    """Calculate Intersection-Over-Union(IOU) of two bounding boxes.
    Parameters
    ----------
    bbox_a : numpy.ndarray
        An ndarray with shape :math:`(N, 4)`.
    bbox_b : numpy.ndarray
        An ndarray with shape :math:`(M, 4)`.
    offset : float or int, default is 0
        The ``offset`` is used to control the whether the width(or height) is computed as
        (right - left + ``offset``).
        Note that the offset must be 0 for normalized bboxes, whose ranges are in ``[0, 1]``.
    Returns
    -------
    numpy.ndarray
        An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
        bounding boxes in `bbox_a` and `bbox_b`.
    """
    if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
        raise IndexError("Bounding boxes axis 1 must have at least length 4")

    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4])

    area_i = np.prod(br - tl + offset, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + offset, axis=1)
    area_b = np.prod(bbox_b[:, 2:4] - bbox_b[:, :2] + offset, axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

def flat_array(pred_scores, match ,gt_difficult):
    n_pos = 0
    score_flat,match_flat = np.array([]),np.array([])
    for match_f, pred_scores_f,gt_difficult_f in zip(match,pred_scores,gt_difficult):
        n_pos += np.logical_not(gt_difficult_f).sum()
        #print(pred_scores_f.reshape(-1).shape)
        score_flat = np.append(score_flat,pred_scores_f.reshape(-1))
        match_flat = np.append(match_flat,match_f.reshape(-1))
    return score_flat,match_flat,n_pos

def average_precision(rec, prec):
    """
    calculate average precision
    Params:
    ----------
    rec : numpy.array
        cumulated recall
    prec : numpy.array
        cumulated precision
    Returns:
    ----------
    ap as float
    """
    if rec is None or prec is None:
        return np.nan

    # append sentinel values at both ends
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], np.nan_to_num(prec), [0.]))

    # compute precision integration ladder
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # look for recall value changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # sum (\delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
def calculate_match(pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels,gt_difficult,iou_thresh = 0.5):
    match = []
    for i in range(len(pred_bboxes)):
        pred_bboxes_f    = pred_bboxes[i]  
        pred_labels_f    = pred_labels[i] 
        pred_scores_f    = pred_scores[i] 
        gt_bboxes_f      = gt_bboxes[i]  
        gt_labels_f      = gt_labels[i]  
        gt_difficult_f   = gt_difficult[i]  
        #print(pred_bboxes_f,pred_labels_f,pred_scores_f,gt_bboxes_f  ,gt_labels_f  )
        #order = pred_score_l.argsort()[::-1]
    
        iou = bbox_iou(pred_bboxes_f,gt_bboxes_f)
        #gt_difficult = np.zeros(gt_bboxes_f.shape[0])
        gt_index = iou.argmax(axis=1)
        #print(gt_index)
        gt_index[iou.max(axis=1) < iou_thresh] = -1
        del iou
    
        selec_f = np.zeros(gt_bboxes_f.shape[0], dtype=bool)
        match_f = np.zeros_like(gt_index)
        for i, gt_idx in enumerate(gt_index):
            if gt_idx >= 0:
                if not selec_f[gt_idx]:
                    match_f[i] = 1
                else:
                    match_f[i] = 0
                selec_f[gt_idx] = True
            else:
                match_f[i] = 0
        match.append(match_f)
    return match

def recall_prec(score_flat,match_flat,n_pos):
    """ get recall and precision from internal records """
    n_fg_class = 1 #max(self._n_pos.keys()) + 1 # we only have one class "mouse" 
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    
    score_l = np.array(score_flat)
    match_l = np.array(match_flat, dtype=np.int32)

    order = score_l.argsort()[::-1]
    match_l = match_l[order]

    tp = np.cumsum(match_l == 1)
    fp = np.cumsum(match_l == 0)
    #print(tp,fp)

    # If an element of fp + tp is 0,
    # the corresponding element of prec[l] is nan.
    with np.errstate(divide='ignore', invalid='ignore'):
        prec = tp / (fp + tp)
    # If n_pos[l] is 0, rec[l] is None.
    if n_pos > 0:
        rec = tp / n_pos
    total_fp = np.sum(match_l==0)
    total_fn = n_pos - np.sum(match_l==1)
    
    return rec, prec,total_fp,total_fn  

def get_mAP(gt_dic,predict_dic):
    #can only do one class right now
    # will implement mutiple object (different mice and objects,food,etc) in the future or if anyone is interested
    pred_bboxes, pred_labels,pred_scores, gt_bboxes,gt_labels,gt_difficult=gt_predicts_numpy(gt_dic,predict_dic)
    match = calculate_match(pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels,gt_difficult)
    score_flat,match_flat,n_pos = flat_array(pred_scores, match ,gt_difficult)
    rec, prec,total_fp,total_fn= recall_prec(score_flat,match_flat,n_pos)
    mean_ap = average_precision(rec,prec)
    return total_fn,total_fp,mean_ap

