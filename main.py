import cv2
import  time
from HandTrakingMod import myHandDetector

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = myHandDetector()

while True:
    _, img = cap.read()
    img = detector.findHands(img)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Winname", img)
    cv2.waitKey(1)
