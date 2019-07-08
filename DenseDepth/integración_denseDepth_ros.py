#!/usr/bin/env python


import sys, time
import numpy as np
from scipy.ndimage import filters
from PIL import Image
import cv2
import roslib
import rospy
from sensor_msgs.msg import CompressedImage
import os
import glob
import argparse
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from loss import depth_loss_function
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt
import cv2
from threading import Thread
from timeit import default_timer as timer
from PIL import Image
import numpy as np
import tensorflow as tf

VERBOSE=False


class image_feature:

    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        self.image_pub = rospy.Publisher("/result/image_raw/compressed",
            CompressedImage,queue_size=0)

        custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}
        # subscribed Topic
        self.denseDepth = model = load_model('kitti.h5', custom_objects=custom_objects, compile=False)
        self.subscriber = rospy.Subscriber("/axis_rgb/camera/image_raw/compressed",
            CompressedImage, self.callback,  queue_size = 1, buff_size=2**24)

        if VERBOSE :
            print ("subscribed to /camera/image/compressed")


    def callback(self, ros_data):
        global graph
        print ("Processing frame | Delay:%6.3f" % (rospy.Time.now() - msg.header.stamp).to_sec())

        start = time.time()
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        if VERBOSE :
            print ('received image of type: "%s"' % ros_data.format)

        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        with graph.as_default():
            result = predict(self.denseDepth, image_np,minDepth=10)
        result = np.asarray(result)[0]
        if VERBOSE:
            cv2.imshow('img',result)
            cv2.imshow('imgNorm',cv2.normalize(result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
            cv2.waitKey(1)
        normalized_result = cv2.normalize(result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', normalized_result)[1]).tostring()
        # Publish new image

        self.image_pub.publish(msg)
        
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
graph = tf.get_default_graph()
rospy.init_node('image_feature', anonymous=True)
ic = image_feature()
try:
	rospy.spin()
except KeyboardInterrupt:
	print ("Shutting down ROS Image feature detector module")

cv2.destroyAllWindows()
