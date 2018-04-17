# -*- coding: utf-8 -*-
# This script isn't the final version of my work of gesture recognization. 
# For that Code Management requires a lot of work, the script is provided merely as reference to show the main process.
# 
# Dataset: RHD_published_v2 [Alternative: NYU handpose dataset/BigHand2.2M]
# 
# Segmentation:
# 	Input:160×160×1 deep_img
# 	output:80×80×1  mask_img
#
# Pose:
# 	Input: image cropped on proposal region
#	Output: 21 scoremap of 2D key poses
#
# Classifier:
#	Input: key part of image cropped on proposal region or 21 key points' location
#	Output:type of gesture

##############################################################################################
##############################################################################################
##                                                                                          ##
##  ##         ##   #####   ###     # #####   ###### ######   #####  ##### ##### ###     #  ##
##  ##    #    ##  #######  # ##    # #   ##  #      ##   ## ##      #     #     # ##    #  ##
##  ##   ###   ## ###   ### #  ##   # #    ## #      ##   ## ##      #     #     #  ##   #  ##
##  ##  ## ##  ## ##     ## #   ##  # #    ## ###### ######   #####  ##### ##### #   ##  #  ##
##  ## ##   ## ## ###   ### #    ## # #    ## #      ## ##        ## #     #     #    ## #  ##
##   ####    ####  #######  #     ### #   ##  #      ##  ##       ## #     #     #     ###  ##
##    ##      ##    #####   #      ## #####   ###### ##   ##  #####  ##### ##### #      ##  ##
##                                                                                          ##
##############################################################################################
##############################################################################################

import tensorflow as tf
import pickle
import os
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append("~/Wonderseen_net/nets")
sys.path.append("~/Wonderseen_Net/utils")
sys.path.append("~/Wonderseen_Net/wonderseen_handpose_fcn/tools")

import cv2
from playsound import playsound
import general
import ReadData
import PostTreatment

# mode
mode = 'predict' # train or predict

# get data
set = 'training'# 'training' 'evaluation'
fatherdic = 'RHD_published_v2/' + set

# Train Para
channel = 1
IMAGE_HEIGHT = 320
IMAGE_WIDTH = 320
trainstep = 50
savestep = 400
start_step = 101200
start_lr = 1e-3
net = general.NetworkOps
saver_restore_addr = '/root/pose-model/handposetemp-model.ckpt-101200'
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH*channel/4], name='INPUT_IMAGE_HEIGHT_MULTI_WIDTH')
realMask = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH*channel/16], name='realMask')
keep_prob = tf.placeholder(tf.float32)

# Classifier Para
CL_graph = tf.Graph()
CLASSIFIER_IMAGE_HEIGHT = 50
CLASSIFIER_IMAGE_WIDTH = 50
HAND_NUM = 1
GESTURE_CLASSES = 17
saver_restore_addr_classifier = '/root/clasiffier-model/handposetemp-model.ckpt-4250'


# write data into memory
if mode == 'train':
    depth_pred = []
    hand_mask_pred = []
    for x in range(0,40000):
        sample_id = random.randint(0,40000)
        # read mask / deep
        mask = scipy.misc.imread(os.path.join(fatherdic, 'mask', '%.5d.png' % sample_id)).astype('float32')
        depth = scipy.misc.imread(os.path.join(fatherdic, 'depth', '%.5d.png' % sample_id)) 
        depth = ReadData.depth_two_uint8_to_float(depth[:, :, 0], depth[:, :, 1])
        depth = cv2.resize(depth,(IMAGE_WIDTH/2,IMAGE_HEIGHT/2))

        print 'load_data',sample_id, x
        mask = cv2.resize(mask,(IMAGE_WIDTH/4,IMAGE_HEIGHT/4)).astype('float32')
        for i in range(0, len(mask)):
            for j in range(0, len(mask[0])):
                if mask[i][j] <= 1:
                    mask[i][j] = 0
                else:
                    mask[i][j] = 1
        all = sum(sum(mask)) + 1e-4
        mask /= all
        depth = depth.reshape(IMAGE_WIDTH // 2 * IMAGE_HEIGHT // 2 * channel)
        depth_pred.append(depth)
        hand_mask_pred.append(mask.reshape(IMAGE_WIDTH//4*IMAGE_HEIGHT//4*channel))

if mode == 'predict':
    pass

# train
def train_handpose_depth_cnn(continueflag):
    global_step = tf.Variable(0, trainable=False)
    add_global = global_step.assign_add(1)
    return_global = global_step.assign(start_step)
    learning_rate = tf.train.exponential_decay(learning_rate = start_lr, global_step=global_step,decay_steps = 10000, decay_rate = 0.97)#,staircase=True)

    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.train.start_queue_runners(sess=sess)

    # Net-Output
    hand_scoremap = depth_handpose_fcn()

    # Loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hand_scoremap, labels=realMask))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Predict
    predict = tf.reshape(hand_scoremap, [-1, IMAGE_HEIGHT, IMAGE_WIDTH])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(realMask, [-1, IMAGE_HEIGHT, IMAGE_WIDTH]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    if continueflag == True:
        saver.restore(sess, saver_restore_addr)
        sess.run([return_global])
    else:
        sess.run(tf.initialize_all_variables())

    # training loop
    lossy = [[],[]]
    plt.figure(figsize=(7,4))
    accuracy = []
    while True:
        step, lr = sess.run([add_global, learning_rate])
        batch_x , batch_y = get_next_data(batch_size=32)
        _, train_loss = sess.run([optimizer, loss], feed_dict={X: batch_x, realMask: batch_y, keep_prob: 0.5})
        if step % trainstep == 0:
            batch_x, batch_y = get_next_data(batch_size=1)
            hand_scoremap1 = sess.run([hand_scoremap], feed_dict={X: batch_x, keep_prob: 1})
            hand_scoremap1 = np.array(hand_scoremap1).reshape(1, 80, 80)
            [batch_x, batch_y] = [np.array(batch_x).reshape(1,160,160), np.array(batch_y).reshape(1,80,80,1)]

            for i in range(0, hand_scoremap1.shape[0]):
                fig = plt.figure(1)
                ax1 = fig.add_subplot('211')
                ax2 = fig.add_subplot('212')
                ax1.imshow(batch_x[i])
                ax2.imshow(hand_scoremap1[i])
                plt.pause(3)

            if step % savestep == 0:
                saver.save(sess, "./mycnnmodel/handposetemp-model.ckpt", global_step=step)
                tf.train.write_graph(sess.graph_def, "./mycnnmodel/","nn_model.pbtxt", False)#as_text=True)

            # simple evaluation on the accuracy of pixel-prediction result
            accuracy.append(cacul_accuracy(hand_scoremap1[0], batch_y[0]))
            print 'accuracy = ', accuracy[-1]
            print 'step,mean-accuracy = ', step, np.mean(accuracy)

        lossy[0].append(step)
        lossy[1].append(train_loss)
        print 'step= ',step, 'train_loss= ',train_loss 
        plt.clf()
        plt.plot(lossy[0], lossy[1], color='blue')
        plt.xlabel('/Step', fontsize=15)
        plt.ylabel('/LOSS', fontsize=15)
        plt.title('FCN Training Loss Iteration', fontsize=18)
        plt.ylim(0, 1.0)
        plt.grid(True, linestyle="-.", color="black", linewidth="1")
        plt.pause(0.01)

# FCN
def depth_handpose_fcn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT/2, IMAGE_WIDTH/2, channel]) 
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 64]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))

    w_c2 = tf.Variable(w_alpha * tf.random_normal([7, 7, 64, 128]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([128]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2,strides=[1, 1, 1, 1], padding='SAME'), b_c2))

    w_c2 = tf.Variable(w_alpha * tf.random_normal([7, 7, 128, 256]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([256]))
    conv2_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c2,strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    maxpool2 = tf.nn.max_pool(conv2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 256, 128]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([128]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(maxpool2, w_c3,
                                                   strides=[1, 1, 1, 1], padding='SAME'), b_c3))

    w_c3_1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 128, 128]))
    b_c3_1 = tf.Variable(b_alpha * tf.random_normal([128]))
    conv3_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, w_c3_1,
                                                   strides=[1, 1, 1, 1], padding='SAME'), b_c3_1))
    dropout3 = tf.nn.dropout(conv3_1, keep_prob)

    w_c3_2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 128, 32]))
    b_c3_2 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv3_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(dropout3, w_c3_2,
                                                   strides=[1, 1, 1, 1], padding='SAME'), b_c3_2))

    w_c4 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 16]))
    b_c4 = tf.Variable(b_alpha * tf.random_normal([16]))
    conv4 = tf.nn.leaky_relu(tf.nn.bias_add(tf.nn.conv2d(conv3_2, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4))
    maxpool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 

    w_f = tf.Variable(w_alpha * tf.random_normal([40*40*16, 80*80*1]))
    b_f = tf.Variable(b_alpha * tf.random_normal([80*80*1]))
    dense = tf.reshape(maxpool4, [-1, w_f.get_shape().as_list()[0]])
    conv_f = tf.nn.leaky_relu(tf.add(tf.matmul(dense, w_f), b_f))
    hand_scoremap = net.fully_connected_relu(conv_f, 'hand_scoremap', 80*80*1)
    return hand_scoremap

# classifier
def gesture_classifier_cnn(w_alpha=0.01, b_alpha=0.1):
    with CL_graph.as_default():
        x = tf.reshape(XX, shape=[-1, CLASSIFIER_IMAGE_HEIGHT, CLASSIFIER_IMAGE_WIDTH, 1]) 
        w_c1 = tf.Variable(w_alpha * tf.random_normal([5, 5, 1, 64]))
        b_c1 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))

        maxpool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        dropout1 = tf.nn.dropout(maxpool1, kkeep_prob)

        w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
        b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(dropout1, w_c2,
                                                       strides=[1, 1, 1, 1], padding='SAME'), b_c2))

        ww_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 128]))
        bb_c3 = tf.Variable(b_alpha * tf.random_normal([128]))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, ww_c3,
                                                       strides=[1, 1, 1, 1], padding='SAME'), bb_c3))

        w_f1 = tf.Variable(w_alpha * tf.random_normal([25 * 25 * 128, 1024]))
        b_f1 = tf.Variable(b_alpha * tf.random_normal([1024]))
        h_f1 = tf.reshape(conv3,[-1,25*25*128])
        h_fc1 = tf.nn.relu(tf.matmul(h_f1,w_f1)+b_f1)
        h_f_drop1 = tf.nn.dropout(h_fc1, kkeep_prob)

        # Fully connected layer
        w_f2 = tf.Variable(w_alpha * tf.random_normal([1024, 170]))
        b_f2 = tf.Variable(b_alpha * tf.random_normal([170]))
        dense = tf.reshape(h_f_drop1, [-1, w_f2.get_shape().as_list()[0]])

        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_f2), b_f2))
        w_out = tf.Variable(w_alpha * tf.random_normal([170, HAND_NUM * GESTURE_CLASSES]))
        b_out = tf.Variable(b_alpha * tf.random_normal([HAND_NUM * GESTURE_CLASSES]))
        out = tf.add(tf.matmul(dense, w_out), b_out)
        return out

with CL_graph.as_default():
    XX = tf.placeholder(tf.float32, [None, CLASSIFIER_IMAGE_HEIGHT * CLASSIFIER_IMAGE_WIDTH], name='INPUT_IMAGE_HEIGHT_MULTI_WIDTH')
    YY = tf.placeholder(tf.float32, [None,  CLASSIFIER_IMAGE_HEIGHT * CLASSIFIER_IMAGE_WIDTH], name='OUTPUT_ONE_HOTS')
    kkeep_prob = tf.placeholder(tf.float32)
    sess1 = tf.Session()
    classifier = gesture_classifier_cnn()
    saver1 = tf.train.Saver()
    saver1.restore(sess1, saver_restore_addr_classifier)

def get_next_data(batch_size = 60):
    depth_pred_batch = []
    hand_mask_pred_batch = []
    for i in range(0,batch_size):
        sample_id = random.randint(0,len(depth_pred)-1)
        depth_pred_batch.append(depth_pred[sample_id])
        hand_mask_pred_batch.append(hand_mask_pred[sample_id])
    return depth_pred_batch, hand_mask_pred_batch


def predict_handscoremap():
    global_step = tf.Variable(0, trainable=False)
    add_global = global_step.assign_add(1)
    return_global = global_step.assign(start_step)
    start_lr = 1e-3 
    learning_rate = tf.train.exponential_decay(learning_rate=start_lr, global_step=global_step, decay_steps=10000,
                                               decay_rate=0.97)  # ,staircase=True)
    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.train.start_queue_runners(sess=sess)
    hand_scoremap = depth_handpose_fcn()

    # Loss
    saver = tf.train.Saver()
    saver.restore(sess, saver_restore_addr)
    while True:
        depth_pre,_ = ReadData.get_one_sample_form_RHD(depth=True,fatherdic=fatherdic)

        # Test
        hand_scoremap1 = sess.run([hand_scoremap], feed_dict={X: depth_pre, keep_prob: 0.5})

        hand_scoremap1 = np.array(hand_scoremap1).reshape(1, 80, 80)
        [depth_pre, hand_scoremap1] = [np.array(depth_pre).reshape(1, 160, 160),
                                       np.array(hand_scoremap1).reshape(1, 80, 80)]
        # upsample
        hand_scoremap1 = cv2.resize(hand_scoremap1[0], (160,160))
        hand_scoremap_cp, hand_scoremap1_show = PostTreatment.eliminate_bkground_from_handscoremap(hand_scoremap1,
                                                                                                   depth_pre,
                                                                                                   threshold=0.25,
                                                                                                   block_half_size=3)
        hand_depth_crop, box = PostTreatment.crop_mask(hand_scoremap_cp, uv_cood_noise = 5, dominate=True)

        hand_depth_crop = cv2.resize(hand_depth_crop,(CLASSIFIER_IMAGE_HEIGHT, CLASSIFIER_IMAGE_WIDTH))
        crop = []
        scale = 1000.
        crop.append(PostTreatment.PreTreatment(hand_depth_crop*scale))
        result = predict_classifier(np.array(crop))

        # Visualization
        plt.close()
        fig = plt.figure(dpi=100,figsize=(10,10))
        ax1 = fig.add_subplot('221')
        import matplotlib.patches as mpatches
        rect = mpatches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='red', linewidth=2)
        ax1.add_patch(rect)
        ax2 = fig.add_subplot('222')
        ax3 = fig.add_subplot('223')
        ax4 = fig.add_subplot('224')
        ax1.imshow(depth_pre[0]+hand_scoremap_cp*10.)
        ax2.imshow(hand_scoremap1_show)
        ax3.imshow(hand_scoremap_cp)
        ax4.imshow(crop[0])

        plt.show()

def predict_classifier(hand_score_crop):
    hand_score_crop = hand_score_crop.reshape(1,2500)
    with CL_graph.as_default():
        predict = tf.reshape(classifier, [-1, HAND_NUM, GESTURE_CLASSES])
        max_idx_p = tf.argmax(predict, axis=2)
        gesture_classifier_result, score = sess1.run([max_idx_p, predict], feed_dict={XX: hand_score_crop, kkeep_prob: 1.})
        print 'predict result：', gesture_classifier_result[0][0], 'score=', score[0,0,int(gesture_classifier_result[0][0])]
    return gesture_classifier_result[0][0]

def cacul_accuracy(hand_scoremap, mask_raw):
    # mask
    max = np.max(hand_scoremap)
    hand_scoremap /= max
    for j in range(0,len(hand_scoremap)):
        for k in range(0,len(hand_scoremap[0])):
            if hand_scoremap[j][k] < 0.8:
                hand_scoremap[j][k] = 0
            else:
                hand_scoremap[j][k] = 1

    # calculate
    accuracy = 0.
    handscore_pre = hand_scoremap.reshape(6400)
    mask_raw = mask_raw.reshape(6400)
    for i in range(0, handscore_pre.shape[0]):
        if handscore_pre[i] == 0. and mask_raw[i] == 0.:
            accuracy += 1.
        if handscore_pre[i] != 0. and mask_raw[i] != 0.:
            accuracy += 1.
    return accuracy/(80*80)


if __name__ == '__main__':
    if mode == 'train':
        train_handpose_depth_cnn(continueflag= True)
    if mode == 'predict':
        predict_handscoremap()
