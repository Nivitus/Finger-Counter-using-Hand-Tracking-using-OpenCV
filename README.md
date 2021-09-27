# Finger-Counter-using-Hand-Tracking-using-OpenCV
To build how to count fingers. We first look into hand tracking and then we will use the hand landmarks to count of the fingers.
# Hand-Tracking-30FPS-using-OpenCV #
## Table of Content ##
- [Overview](#overview) 
- [Motivation](#motivation) 
- [Installation](#installation) 
- [About the MediaPipe](#about-the-mediapipe) 
- [Libraries Used](#libraries-used) 
- [Code Snippets](#code-snippets)
- [Deployment](#deployment)  
- [Team](#team) 

## Overview 
OpenCV is a library used for computer vision applications. With help of OpenCV, we can build an enormous number of applications that work better in real-time. Mainly it is used for image and video processing. This project all about Hand Tracking in real time using OpenCV. In this Hand Tracking project, we’ve built a hand gesture recognizer using OpenCV and python. We’ve used MediaPipe and Tensorflow framework for the detection and gesture recognition respectively. Here we’ve learned about the basics of the Neural Network, File handling, some common image processing techniques, etc.

![Hand_Tracking](Images/fii.gif)

## Motivation 
The motivation for developing computer vision projects is the human vision system which is richest sense that we have. I heard about omplex cameras are being integrated into many devices (Smartphones how have cameras producing depth images as well as colour images, wide angle cameras are being integrated into cars so that a birds-eye image can be created, cameras are appearing in smart glasses) and this in turn is pushing the development of progressively more complex vision systems. SO that is big mmotivation to me for learning opencv and basics of computer vision projects.

## Installation 
The Code is written in Python 3.7. If you don't have Python installed just [clik here](https://www.python.org/downloads/) and install python on your system. 
If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, such as opencv, numpy and mediapipe that's it

``` python

Install the required modules

–> pip install opencv-python
-> pip install numpy
–> pip install mediapipe
```

## About the MediaPipe

MediaPipe is a framework mainly used for building audio, video, or any time series data. With the help of the MediaPipe framework, we can build very impressive pipelines for different media processing functions.

Some of the major applications of MediaPipe.

* Multi-hand Tracking
* Face Detection
* Object Detection and Tracking
* Objectron: 3D Object Detection and Tracking
* AutoFlip: Automatic video cropping pipeline etc.

![Ronald Fisher](Images/mediapipe.png)


## Libraries Used 

[![Alt Text](Images/mo.JPG)]

## Code Snippets

``` python

# Install the Libraries

import cv2
import mediapipe as mp
import time

```
``` python

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
cTime = 0
pTime = 0
```
``` python
# Function Start
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                #if id == 5:
                cv2.circle(img, (cx, cy), 15, (139, 0, 0), cv2.FILLED)


            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
```
``` python 
# Time and FPS Calculation

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (139,0,0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

```
## Deployment 

There is no Deployment now, maybe in the future this project definitely gonna deploy.

## Team

![](Images/nivi.JPG)





