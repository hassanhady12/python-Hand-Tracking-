import cv2
import mediapipe as mp
import time

class myHandDetector:
    def __init__(self, static_mod=False, max_num_hands=2, min_det_conf=0.5, min_track_conf=0.5):
        self.static_mod = static_mod
        self.max_num_hands = max_num_hands
        self.min_det_conf = min_det_conf
        self.min_track_conf = min_track_conf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)

        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo = 0, draw = True):

        lmLists = []

        if self.result.multi_hand_landmarks:
            self.result.multi_hand_landmarks[handNo]
            for handLms in self.result.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    # print(id, lm)

                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id, cx, cy)
                    lmLists.append([id, cx, cy])
                    if draw:
                        self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return lmLists

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = myHandDetector()

    while True:
        _, img = cap.read()
        img = detector.findHands(img)
        mlList = detector.findPosition(img, draw=False)
        if len(mlList) > 0:
             print(mlList[0])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Winname", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
