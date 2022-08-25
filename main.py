import itertools
import math
import os
import numpy as np
import yaml
import cv2
import utils

cmMinDist = 100
cmCalibDist = 180
birdScale = 1/3.5

outputPath = './output/'
weightsPath = "./yolo/yolov3.weights"
configPath = "./yolo/yolov3.cfg"
confid = 0.5
thresh = 0.5

configName = input("Inserisci il nome del file di configurazione: ")
with open("./conf/" + configName, "r") as ymlfile:
    videoData = yaml.full_load(ymlfile)
videoName = os.path.splitext(configName)[0]

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
prefix = videoData['prefix']

M, birdImage = utils.birdPerspectiveTransform(cornerPoints, W, H, cv2.imread(imgPath))

pxMinDist = utils.getPxMinDist(cmMinDist, cmCalibDist, calibDistPoint, M)
pxWarnDist = pxMinDist * 1.2

vs = cv2.VideoCapture(videoPath)
fps = int(vs.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*"XVID")
outputMovie = cv2.VideoWriter(outputPath + prefix + ".avi", fourcc, fps, (W, H))

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
        status = np.zeros(len(birdPoints))
        tempBird = birdImage.copy()

        pairsInfo = []

        listIndexes = list(itertools.combinations(range(len(birdPoints)), 2))
        for i, pair in enumerate(itertools.combinations(birdPoints, 2)):
            if utils.inROI(pair[0], W, H) and utils.inROI(pair[1], W, H):
                idx1 = listIndexes[i][0]
                idx2 = listIndexes[i][1]
                dist = math.dist(pair[0], pair[1])
                if dist <= pxMinDist:
                    status[idx1] = 1
                    status[idx2] = 1
                    pairsInfo.append([idx1, idx2, 1])
                elif pxMinDist < dist <= pxWarnDist:
                    if status[idx1] != 1:
                        status[idx1] = 2
                    if status[idx2] != 1:
                        status[idx2] = 2
                    pairsInfo.append([idx1, idx2, 2])

        for index, birdPoint in enumerate(birdPoints):
            if utils.inROI(birdPoint, W, H):
                tempBird = utils.printCircle(tempBird, birdPoint, status[index])
                frame = utils.printRectangle(frame, filteredBoxes[index], status[index])

        for info in pairsInfo:
            i = info[0]
            j = info[1]
            status = info[2]
            frame = utils.printLine(frame, filteredBoxes[i], filteredBoxes[j], status)
            tempBird = utils.printBirdLine(tempBird, birdPoints[i], birdPoints[j], status)

        pnts = np.array(cornerPoints, np.int32)
        cv2.polylines(frame, [pnts], True, (0, 0, 0), thickness=1)

        result = frame.copy()
        sW = int(W * birdScale)
        sH = int(H * birdScale)
        tempBird = cv2.resize(tempBird, (sW, sH))
        tempBird = cv2.rotate(tempBird, cv2.ROTATE_90_CLOCKWISE)
        result[H - sW:, :sH] = tempBird

        cv2.imshow('Result', result)
        outputMovie.write(result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

vs.release()
