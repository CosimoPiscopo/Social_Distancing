import os
import cv2
import yaml

mouse_pts = []
DOT_SIZE = 5
CORNER_DOT_COLOR = LINE_COLOR = (0, 0, 0)
DISTANCE_DOT_COLOR = DISTANCE_LINE_COLOR = (255, 0, 0)
LINE_THICK = 1
WINDOW_NAME = 'CalibImage'


def mouseCallback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pts.append((x, y))
        if len(mouse_pts) <= 4:
            cv2.circle(frame, (x, y), DOT_SIZE, CORNER_DOT_COLOR, -1)
            if len(mouse_pts) != 1:
                cv2.line(frame, (x, y), (mouse_pts[len(mouse_pts) - 2][0], mouse_pts[len(mouse_pts) - 2][1]),
                         LINE_COLOR, LINE_THICK)
            if len(mouse_pts) == 4:
                cv2.line(frame, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), LINE_COLOR, LINE_THICK)
        else:
            cv2.circle(frame, (x, y), DOT_SIZE, DISTANCE_DOT_COLOR, -1)
            if len(mouse_pts) == 6:
                cv2.line(frame, (x, y), (mouse_pts[4][0], mouse_pts[4][1]), DISTANCE_LINE_COLOR, LINE_THICK)


videoPath = './video/'
imgPath = './img/'
configPath = './conf/'

videoName = input('Inserisci il nome del video: ')
prefix = os.path.splitext(videoName)[0]
configName = prefix + '.yml'
imgName = prefix + '.jpg'

vs = cv2.VideoCapture(videoPath + videoName)

grabbed, frame = vs.read()

if not grabbed:
    print('Calibrazione non riuscita!')
    vs.release()
    exit()

cv2.imwrite(imgPath + imgName, frame)
H, W = frame.shape[:2]

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, mouseCallback)

while True:
    cv2.imshow(WINDOW_NAME, frame)

    if len(mouse_pts) == 7:
        cv2.destroyWindow(WINDOW_NAME)
        config_data = dict(
            tl=mouse_pts[0],
            tr=mouse_pts[1],
            br=mouse_pts[2],
            bl=mouse_pts[3],
            d1=mouse_pts[4],
            d2=mouse_pts[5],
            width=W,
            height=H,
            prefix=prefix,
            videoPath=videoPath + videoName,
            imgPath=imgPath + imgName
        )
        with open(configPath + configName, 'w') as outfile:
            yaml.dump(config_data, outfile, default_flow_style=False)
        break

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
