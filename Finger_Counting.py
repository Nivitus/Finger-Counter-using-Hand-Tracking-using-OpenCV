import cv2
import os
import time
import Hand_Tracking_Module as htm


cap = cv2.VideoCapture(0)
wCam = 640
hCam = 480

cap.set(3, wCam)
cap.set(4, hCam)

FolderPath = "FingerImages"
MyList = os.listdir(FolderPath)
print(MyList)
overlayList = []
for imPath in MyList:
    image = cv2.imread(f'{FolderPath}/{imPath}')
    #print(f'{FolderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))
pTime = 0


detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:

    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)
    #print(lmlist)

    if len(lmlist) != 0:
        fingers = []

        #Thumps

        if lmlist[tipIds[0]][1] > lmlist[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Four Fingers
        for id in range(1, 5):
            if lmlist[tipIds[id]][2] < lmlist[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)


        # Printing Number of Fingers

        totalFingers = fingers.count(1)
        print(totalFingers)


        h, w, c = overlayList[totalFingers-1].shape
        img[0:h, 0:w] = overlayList[totalFingers-1]

        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (400, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (220, 20, 60), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
