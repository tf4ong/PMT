from absl import app, flags, logging
from absl.flags import FLAGS
import os
import shutil
import tensorflow as tf
from core.yolov4 import YOLO, decode, compute_loss, decode_train
from core.dataset import Dataset
from core.config import cfg
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all
from prt_utils.configloader import train_config_loader


#flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
#flags.DEFINE_string('weights', './scripts/yolov4.weights', 'pretrained weights')
#flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('config_path',  'C:/PRT/configs/config.ini', 'location of config file')
def train(ts):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #if len(physical_devices) > 0:
    #    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    trainset = Dataset(ts['model'],ts['classes'],ts['trainpath_txt'],ts['size'],ts['train_batch_size'],ts['train_data_aug'])
    #testset = Dataset(ts['model'],ts['classes'],ts['trainpath_txt'],ts['size'],ts['test_batch_size'],ts['test_data_aug'])
    logdir = "./data/log"
    isfreeze = False
    steps_per_epoch = len(trainset)
    first_stage_epochs = int(ts['FISRT_STAGE_iteration']/steps_per_epoch)
    second_stage_epochs = int(ts['SECOND_STAGE_iteration']/steps_per_epoch)
    if not ts['transfer_learning']:
        first_stage_epochs = 0 
    else:
        first_stage_epochs = first_stage_epochs#cfg.TRAIN.FISRT_STAGE_EPOCHS
    #second_stage_epochs = ts['SECOND_STAGE_EPOCHS']#cfg.TRAIN.SECOND_STAGE_EPOCHS
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = ts['train_warmup_iteration']
    total_steps =  ts['FISRT_STAGE_iteration']+ ts['SECOND_STAGE_iteration']
    #total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch
    # train_steps = (first_stage_epochs + second_stage_epochs) * steps_per_period

    input_layer = tf.keras.layers.Input([ts['size'], ts['size'], 3])
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(ts['model'],ts['classes'])
    IOU_LOSS_THRESH = ts['iou_loss_thresh']

    freeze_layers = utils.load_freeze_layer(ts['model'], ts['tiny'])

    feature_maps = YOLO(input_layer, NUM_CLASS,ts['model'], ts['tiny'])
    if  ts['tiny']:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, ts['size'] // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, ts['size'] // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
    else:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, ts['size'] // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            elif i == 1:
                bbox_tensor = decode_train(fm, ts['size'] // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, ts['size'] // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model.summary()

    if not ts['transfer_learning']:
        print("Training from scratch")
    else:
        if ts['weights'].split(".")[len(ts['weights'].split(".")) - 1] == "weights":
            utils.load_weights(model,ts['weights'], ts['model'], ts['tiny'])
        else:
            model.load_weights(ts['weights'])
        print('Restoring weights from: %s ... ' % ts['weights'])

    optimizer = tf.keras.optimizers.Adam()
    if os.path.exists(logdir): shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir)

    # define training step function
    # @tf.function
    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            tf.print("=> STEP %4d/%4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, total_steps, optimizer.lr.numpy(),
                                                               giou_loss, conf_loss,
                                                               prob_loss, total_loss))
            # update learning rate
            global_steps.assign_add(1)
            if global_steps < warmup_steps:
                lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
            else:
                lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                )
            optimizer.lr.assign(lr.numpy())

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()
            return global_steps, optimizer.lr, total_loss, giou_loss, conf_loss, prob_loss
    def test_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            tf.print("=> TEST STEP %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, giou_loss, conf_loss,
                                                               prob_loss, total_loss))
            return global_steps, optimizer.lr, total_loss, giou_loss, conf_loss, prob_loss
    iteration = 0
    logFile_train = open(f"{ts['outpath']}/train_loss.csv", 'w', encoding="utf-8")
    logFile_train.write( 'iteration,global_step,lr,total_loss,giou_loss,conf_loss,prob_loss\n')
    #logFile_test = open(f"{ts['outpath']}/test_loss.csv", 'w', encoding="utf-8")
    #logFile_test.write( 'iteration,epoch,global_step,lr,total_loss,giou_loss,conf_loss,prob_loss\n')
    while True:
        for image_data, target in trainset:
            if iteration < ts['FISRT_STAGE_iteration']:
                if not isfreeze:
                    isfreeze = True
                    for name in freeze_layers:
                        freeze = model.get_layer(name)
                        freeze_all(freeze)
            elif iteration >= ts['SECOND_STAGE_iteration']:
                if isfreeze:
                    isfreeze = False
                    for name in freeze_layers:
                        freeze = model.get_layer(name)
                        unfreeze_all(freeze)
            lost_train=train_step(image_data, target)
            logFile_train.write(f'{iteration},{lost_train[0].numpy()},{lost_train[1].numpy()},{lost_train[2]},{lost_train[3]},{lost_train[4]},{lost_train[5]}\n')

            if iteration%ts['save_iteration'] ==0 and iteration>0:
                if not os.path.exists(f"{ts['outpath']}/{iteration}"):
                    os.mkdir(f"{ts['outpath']}/{iteration}")
                model.save_weights(f"{ts['outpath']}/{iteration}/weight")
                #model.save(f"{ts['outpath']}/{iteration}/weight.h5")
            if iteration > (ts['FISRT_STAGE_iteration'] + ts['SECOND_STAGE_iteration']):
                break
            iteration+=1
        if iteration > (ts['FISRT_STAGE_iteration'] + ts['SECOND_STAGE_iteration']):
            break
    lost_train.close()
    '''
    for iter in range(ts['FISRT_STAGE_iteration'] + ts['SECOND_STAGE_iteration']):
        if iter < ts['FISRT_STAGE_iteration']:
            if not isfreeze:
                isfreeze = True
                for name in freeze_layers:
                    freeze = model.get_layer(name)
                    freeze_all(freeze)
        elif iter >= ts['SECOND_STAGE_iteration']:
            if isfreeze:
                isfreeze = False
                for name in freeze_layers:
                    freeze = model.get_layer(name)
                    unfreeze_all(freeze)
        for image_data, target in trainset:
            lost_train=train_step(image_data, target)
            logFile_train.write(f'{iteration},{lost_train[0].numpy()},{lost_train[1].numpy()},{lost_train[2]},{lost_train[3]},{lost_train[4]},{lost_train[5]}\n')
        #for image_data, target in testset:
        #    lost_test=test_step(image_data, target)
        #    logFile_test.write(f'{epoch},{lost_test[0].numpy()},{lost_test[1].numpy()},{lost_test[2]},{lost_test[3]},{lost_test[4]},{lost_test[5]}\n')
        if iter%ts['save_epoch'] ==0:
            if not os.path.exists(f"{ts['outpath']}/{iteration}"):
                os.mkdir(f"{ts['outpath']}/{iteration}")
            model.save_weights(f"{ts['outpath']}/{iteration}/weight")
        iteration+=1
    '''
    #lost_train.close()
    #lost_test.close()

def main(_argv):
    train(FLAGS.config_path)
    return

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass