import cv2
from cvzone.HandTrackingModule import HandDetector


cap = cv2.VideoCapture(0)
detectors = HandDetector(detectionCon=0.8, maxHands=2)
while True:
    success, img = cap.read()
    hands, img = detectors.findHands(img)   # with draw which hand it is

    if hands:
        # hand1
        hand1 = hands[0]
        lmList = hand1["lmList"]  # list of all 21 point of the hands
        bbox = hand1["bbox"]  # giving bounding box info: x, y, w,h
        centerPoint1 = hand1["center"]  # giving us x,y point of the mid of the bounding box
        handType1 = hand1["type"]  # gives us types of the hands

        if len(hands)==2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            bbox2 = hand2["bbox"]
            centerPoint2 = hand2["center"]
            handType2 = hand2["type"]
            # length, info, img = detectors.findDistance(centerPoint1, centerPoint2, img)
            #print(length)
            length, info, img = detectors.findDistance(lmList[4], lmList2[13], img)
            length, info, img = detectors.findDistance(lmList[20], lmList2[12], img)
            length, info, img = detectors.findDistance(lmList[12], lmList2[1], img)
            length, info, img = detectors.findDistance(lmList[1], lmList2[13], img)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

