import math
import numpy as np
import cv2


COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
BIG_CIRCLE = 30
SMALL_CIRCLE = 5


def birdPerspectiveTransform(cornerPoints, W, H, image):
    src = np.float32(cornerPoints)
    dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    M = cv2.getPerspectiveTransform(src, dst)
    birdImage = np.zeros(
        (int(H), int(W), 3), np.uint8
    )
    birdImage[:] = (0, 0, 0)
    # birdImage = cv2.warpPerspective(image, M, (W, H))
    return M, birdImage


def getTransformedGroundPoints(boxes, M):
    birdPoints = []
    for box in boxes:
        pnt = [int(box[0] + (box[2] * 0.5)), int(box[1] + box[3])]
        t_pnt = getTransformedPoint(pnt, M)
        birdPoints.append(t_pnt)
    return birdPoints


def getTransformedPoint(pnt, M):
    src = np.float32([[pnt]])
    dst = cv2.perspectiveTransform(src, M)[0][0]
    return dst


def getPxMinDist(cmMinDist, cmCalibDist, pnts, M):
    t_pnt1 = getTransformedPoint(pnts[0], M)
    t_pnt2 = getTransformedPoint(pnts[1], M)
    pxMinDist = math.dist(t_pnt1, t_pnt2) * (cmMinDist / cmCalibDist)
    return pxMinDist


def inROI(birdPoints, boxes, W, H):
    filtBirdPnts = []
    filtBox = []
    for i, pnt in enumerate(birdPoints):
        if 0 < pnt[0] < W and 0 < pnt[1] < H:
            filtBirdPnts.append(pnt)
            filtBox.append(boxes[i])
    return filtBirdPnts, filtBox


def statusColor(status):
    match status:
        case 0:
            color = COLOR_GREEN
        case 1:
            color = COLOR_RED
        case 2:
            color = COLOR_YELLOW
    return color


def printCircle(frame, centerPnt, status):
    color = statusColor(status)
    cv2.circle(frame, centerPnt.astype(int), BIG_CIRCLE, color, 2)
    cv2.circle(frame, centerPnt.astype(int), SMALL_CIRCLE, color, -1)
    return frame


def printRectangle(frame, box, status):
    color = statusColor(status)
    cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2)
    return frame


def printLine(frame, box1, box2, status):
    color = statusColor(status)
    x, y, w, h = box1[:]
    x1, y1, w1, h1 = box2[:]
    frame = cv2.line(frame, (int(x + w / 2), int(y + h / 2)), (int(x1 + w1 / 2), int(y1 + h1 / 2)), color, 2)
    return frame


def printBirdLine(frame, pnt1, pnt2, status):
    color = statusColor(status)
    frame = cv2.line(frame, pnt1.astype(int), pnt2.astype(int), color, 2)
    return frame
