import itertools
import math

import numpy as np
import yaml
import cv2
import utils

cmMinDist = 100
cmCalibDist = 180

outputPath = './output/'
weightsPath = "./yolo/yolov3.weights"
configPath = "./yolo/yolov3.cfg"
confid = 0.5
thresh = 0.5

configName = input("Inserisci il nome del file di configurazione: ")
configName = 'vid_short.yml'
with open("./conf/" + configName, "r") as ymlfile:
    videoData = yaml.full_load(ymlfile)

cornerPoints = list()
cornerPoints.append(videoData['tl'])
cornerPoints.append(videoData['tr'])
cornerPoints.append(videoData['br'])
cornerPoints.append(videoData['bl'])

calibDistPoint = [videoData['d1'], videoData['d2']]

W = videoData['width']
H = videoData['height']
videoPath = videoData['videoPath']
imgPath = videoData['imgPath']

M, birdImage = utils.birdPerspectiveTransform(cornerPoints, W, H, cv2.imread(imgPath))

pxMinDist = utils.getPxMinDist(cmMinDist, cmCalibDist, calibDistPoint, M)
pxWarnDist = pxMinDist * 1.2

vs = cv2.VideoCapture(videoPath)
fps = int(vs.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*"XVID")
outputMovie = cv2.VideoWriter(outputPath + "distancing.avi", fourcc, fps, (W, H))
birdMovie = cv2.VideoWriter(outputPath + "birdEye.avi", fourcc, fps, (W, H))

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
outputLayers = net.getUnconnectedOutLayersNames()

while True:

    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(outputLayers)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:

        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == 0:

                if confidence > confid:
                    centerX, centerY, boxWidth, boxHeight = detection[:4] * np.array(
                        [W, H, W, H])  # Parameters of the bounding box are normalized.

                    x = int(centerX - (boxWidth / 2))
                    y = int(centerY - (boxHeight / 2))

                    boxes.append([x, y, int(boxWidth), int(boxHeight)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)

    if len(indices) > 0:

        filteredBoxes = []
        for i in indices:
            filteredBoxes.append(boxes[i])

        birdPoints = utils.getTransformedGroundPoints(filteredBoxes, M)
        tempBird = birdImage.copy()
        for point in birdPoints:
            tempBird = utils.printCircle(tempBird, point, 0)

        if len(birdPoints) >= 2:
            for index, birdPoint in enumerate(birdPoints):
                if utils.inROI(birdPoint, W, H):
                    frame = utils.printRectangle(frame, filteredBoxes[index], 0)

            listIndexes = list(itertools.combinations(range(len(birdPoints)), 2))
            for i, pair in enumerate(itertools.combinations(birdPoints, 2)):
                if utils.inROI(pair[0], W, H) and utils.inROI(pair[1], W, H):
                    dist = math.dist(pair[0], pair[1])
                    if pxMinDist < dist <= pxWarnDist:
                        tempBird = utils.printCircle(tempBird, pair[0], 2)
                        tempBird = utils.printCircle(tempBird, pair[1], 2)

                        i1 = listIndexes[i][0]
                        i2 = listIndexes[i][1]

                        frame = utils.printRectangle(frame, filteredBoxes[i1], 2)
                        frame = utils.printRectangle(frame, filteredBoxes[i2], 2)
                        frame = utils.printLine(frame, filteredBoxes[i1], filteredBoxes[i2], 2)

            for i, pair in enumerate(itertools.combinations(birdPoints, 2)):
                dist = math.dist(pair[0], pair[1])
                if utils.inROI(pair[0], W, H) and utils.inROI(pair[1], W, H):
                    dist = math.dist(pair[0], pair[1])
                    if dist <= pxMinDist:
                        tempBird = utils.printCircle(tempBird, pair[0], 1)
                        tempBird = utils.printCircle(tempBird, pair[1], 1)

                        i1 = listIndexes[i][0]
                        i2 = listIndexes[i][1]

                        frame = utils.printRectangle(frame, filteredBoxes[i1], 1)
                        frame = utils.printRectangle(frame, filteredBoxes[i2], 1)
                        frame = utils.printLine(frame, filteredBoxes[i1], filteredBoxes[i2], 1)

        pnts = np.array(cornerPoints, np.int32)
        cv2.polylines(frame, [pnts], True, (0, 0, 0), thickness=1)

        outputMovie.write(frame)
        birdMovie.write(tempBird)

        cv2.imshow('Standard View', frame)
        cv2.imshow('Bird View', tempBird)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

vs.release()
