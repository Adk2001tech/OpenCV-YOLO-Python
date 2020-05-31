
#import libraries
import os
os.chdir(r'D:/projects/Computer-Vision-with-Python-opencv/06-Deep-Learning-Computer-Vision/YOLO_v3')
import numpy as np
import cv2
from model.yolo_model import YOLO
import matplotlib.pyplot as plt
#-------

#Extracting classes names from coco datasets

with open('data/coco_classes.txt') as f:
    classes=f.readlines()
coco_classes=[c.strip() for c in classes] 
#-------

# Pre-processing image to feed in our YOLO model
def img_setup(img):
    pre_img=cv2.resize(img,(416,416))
    pre_img=pre_img.astype('float32')
    pre_img/=255.0
    pre_img=np.expand_dims(pre_img,axis=0)
    return pre_img
#--------


# Feeding image data into YOLO model
# Extracting boxes co-ordinates, index no. of class, accuracies of each classes
# Drawing rectangles with respective classes name + accuracy
def detect(img,classes,yolo):
    feed_img=img_setup(img)
    boxes,num_class,accuracy=yolo.predict(feed_img, img.shape)
    for box,num,acc in zip(boxes,num_class,accuracy):
        x,y,w,h=tuple(map(int,box))
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
        cv2.putText(img,classes[num]+':'+str(acc),(x,y),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    1,(0,255,0),3,cv2.LINE_AA)
    return img 
#------


#Loading Model
#1st parameter: %Threshold for object detection
#2nd parameter: %Threshold for box detection
yolo=YOLO(0.5,0.5)
#-----

#Display Video
cap=cv2.VideoCapture('D:/projects/highway.mp4')
_,frame=cp.read()

plt.imshow(detect(frame))
while True:
    _,frame=cap.read()
    frame=detect(frame,coco_classes,yolo)
    cv2.imshow('obj_detection',frame)
    if cv2.waitKey(30) & 0xff==27:
        break
cap.release()
cv2.destroyAllWindows()
#----