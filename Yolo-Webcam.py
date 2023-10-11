from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture(0)

model = YOLO("../Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# gates = [0, 320, 640, 320] horizontal gates
gates = [320, 80, 320, 380]

enterCount = []
outCount = []

allID = {}

while True:
    success, img = cap.read()

    # imgRegion = cv2.bitwise_and(img, mask)

    results = model(img, stream=True)
    # results = model(imgRegion, stream=Ture)

    detections = np.empty((0, 5))

    # Draw Gates
    cv2.line(img,(gates[0], gates[1]),
             (gates[2], gates[3]),(0, 0, 255), thickness=3)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100  # Object %
            # Class Name
            cls = int(box.cls[0])

            currentClass = classNames[cls]  # Detect class of object
            if currentClass == 'person' and conf > 0.3:

                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack([detections, currentArray])  # Add person's coordinates

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        Id = str(Id)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 30, 0))
        cvzone.putTextRect(img, f'{Id}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=2, offset=5)

        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx, cy), 5, (255, 50, 20), cv2.FILLED)

        if Id not in allID:
            allID[Id] = cx

        if (gates[1] < cy < gates[3] and ((gates[0] - 20) < cx < (gates[2] + 20))
                and (Id not in outCount and Id not in enterCount)):

            if allID[Id] < gates[2]:
                outCount.append(Id)
            else:
                enterCount.append(Id)

    print(len(outCount), "OUT")
    print(len(enterCount), "IN")
    cv2.imshow("Image", img)

    cv2.waitKey(1)
