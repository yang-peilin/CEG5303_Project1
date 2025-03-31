The yolov7 project is a basic implementation. There are three data sets in VOCdevkit: VOC2007 with 20 classes, a subset of VOC2007 data set "small" with 8 classes selected, and a custom data set for experiment with "robot" class added to the 8 classes for implementation;
voc2yolo.py, voc2yolo_small.py, voc2yolo_robot.py are the script codes for converting the three xml files into txt files

yolov7_adv is a project that implements attack and defense. It contains attack function, and models/common.py also replaces ReLU with ISReLU function
