import tensorflow as tf
import matplotlib.image as mpimg
import keras_preprocessing
from keras_preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
from numpy import loadtxt
import cv2
import sys
import imutils
import os
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
model=load_model('rps.h5')
i=0
decision="a"
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,


        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    #frame = cv2.resize(cv2.imread(frame), (150, 150))#.astype(np.float32)




    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.putText(frame,decision,(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(255, 0, 0),3)



    cv2.imshow('Video', frame)

    if(len(faces)>0):
     print(faces[0][2])
     crop_img = frame[faces[0][1]:faces[0][1]+faces[0][3],faces[0][0]:faces[0][0]+faces[0][2]]
     frame = cv2.resize(crop_img, (150,150),interpolation=cv2.INTER_AREA)
     cv2.imwrite('pic'+str(i)+'.jpg',frame)
     print(faces)

