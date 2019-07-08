# TFG Francisco Javier Gomez Sanchez
## Ingeniería Robótica, Electrónica y Mecatrónica

---

## How to use

### Required libraries
* ROS melodic
* Python 3.6
* tensorflow-gpu==1.13.1
* Pillow==6.0.0
* Keras==2.2.4
* opencv-python==4.1.0.25


### To run yolov3 on the webcam

* Connect the webcam
* run: `python3 keras-yolo3/yolo.py`

### To run denseDepth on the webcam

* Connect the webcam
* run: `python3 DenseDepth/test.py`

### Process the rosbag with yolov3

* run `roscore`
* run `rosbag play -l 2018-06-01-12-07-14.bag`
* run `python3 keras-yolo3/integration_yolo_ros.py`
* run `rviz`

### Process the rosbag with denseDepth

* run `roscore`
* run `rosbag play -l 2018-06-01-12-07-14.bag`
* run `python3 DenseDepth/integración_denseDepth_ros.py`
* run `rviz`

### Process the rosbag with yolov3 and deepdepth
* run `roscore`
* run `rosbag play -l 2018-06-01-12-07-14.bag`
* run `python3 integration_yolo_denseDepth/integration_yolo_dense_depth_ros.py`
* run `rviz`