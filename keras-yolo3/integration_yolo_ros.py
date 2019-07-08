#!/usr/bin/env python


import sys, time
import numpy as np
from PIL import Image
import cv2
import roslib
import rospy
from sensor_msgs.msg import CompressedImage
from yolo import YOLO
import tensorflow as tf

VERBOSE=False


class pedestrian_detection:

    def __init__(self):
        '''Load model, initialize ros publisher and ros subscriber'''
        # load model
        self.yolo = YOLO()
        self.subscribe_topic = "/axis_rgb/camera/image_raw/compressed"
        self.publish_topic = "/result/image_raw/compressed"
        # topic where we publish
        self.image_pub = rospy.Publisher(self.publish_topic,
            CompressedImage,queue_size=1, buff_size=2**24)


        # subscribed Topic
        self.subscriber = rospy.Subscriber(self.subscribe_topic,
            CompressedImage, self.callback,  queue_size = 1)

        if VERBOSE :
            print ("subscribed to {}".format(self.subscribe_topic))


    def callback(self, ros_data):
        global graph
        '''Callback function of subscribed topic. 
        Here the model predicts on each image and publish the result'''
        
        # read the image from the topic, cv_bridge could not be used here 
        # because of the compression
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # convert to PIL format
        image = Image.fromarray(image_np)

        # use default graph due to multiprocessing issues in Keras
        with graph.as_default():
            # predict with yolo3
            image = self.yolo.detect_image(image)

        # convert result to numpy array
        result = np.asarray(image)

        if VERBOSE:
            # plot the result
            cv2.imshow('img',result)
            cv2.waitKey(1)
        
        # publish the image with the annotations
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', result)[1]).tostring()
        self.image_pub.publish(msg)
        





# check if the GPU is loaded
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# save default graph
graph = tf.get_default_graph()

# init rospy node
rospy.init_node('image_feature', anonymous=True)
ic = pedestrian_detection()
try:
	rospy.spin()
except KeyboardInterrupt:
	print ("Shutting down ROS YOLO integration")
cv2.destroyAllWindows()

