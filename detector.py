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


    # Display the resulting frame



     img = image.load_img('pic'+str(i)+'.jpg', target_size=(150, 150))

     x1 = image.img_to_array(img)
     x1 = np.expand_dims(x1, axis=0)
     images = np.vstack([x1])
     classes = model.predict(images)
     print(classes)
     classes[0][0]=round(classes[0][0])
     classes[0][1]=round(classes[0][1])
     print(classes)
     if(classes[0][0]==1):
      decision="with mask"
      print("with mask")
     else:
      decision="without mask"
      print("without mask")
     os.remove('pic'+str(i)+'.jpg')
     i=i+1


    else:
     frame = cv2.resize(frame, (150,150),interpolation=cv2.INTER_AREA)


    # Display the resulting frame

     cv2.imwrite('pic'+str(i)+'.jpg',frame)

     img = image.load_img('pic'+str(i)+'.jpg', target_size=(150, 150))

     x1 = image.img_to_array(img)
     x1 = np.expand_dims(x1, axis=0)
     images = np.vstack([x1])
     classes = model.predict(images)
     print(classes)
     classes[0][0]=round(classes[0][0])
     classes[0][1]=round(classes[0][1])
     print(classes)
     if(classes[0][0]==1):
      decision="with mask"
      print("with mask")
     else:
      decision="without mask"
      print("without mask")
     os.remove('pic'+str(i)+'.jpg')
     i=i+1

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break


video_capture.release()
cv2.destroyAllWindows()




#filepath="/home/rahul/rps.h5"


##img = image.load_img('/home/rahul/Desktop/Bump problem/rps-test-set/with_mask/106-with-mask.jpg', target_size=(150, 150))


###In the program ########

#There are two if conditions, whicha are meant for cropping the image with the rectangle around the face.
# and else condition which takes the whole image.
#The else condition is used as the frontal_face detector is not that capable of detecting face when it is masked.
#So there we have used the general whole image .
#Overall this cropping have helped in the icreasing the accuracy as only the face is used for or trained NN model prediction.
