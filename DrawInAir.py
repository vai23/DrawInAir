import cv2
import numpy as np
from collections import deque, namedtuple

color = namedtuple('color', ['red', 'green', 'blue', 'lblue', 'white', 'purple'])
color = color([0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 102, 102], [255, 255, 255], [255, 102, 153])
points = [deque([deque(maxlen=1024)]), deque([deque(maxlen=1024)]), deque([deque(maxlen=1024)])]
lowers = None
uppers = None


def drawOptions(image):
    global color

    shape = (100, 100, 3)

    clearAll = np.zeros(shape, np.uint8) + color.lblue
    cv2.putText(clearAll, 'Clear All', (10, 55), cv2.FONT_HERSHEY_COMPLEX, 0.5, color.white)
    image[0:100, 0:100] = clearAll

    red = np.zeros(shape, np.uint8) + color.red
    cv2.putText(red, 'Red', (30, 55), cv2.FONT_HERSHEY_COMPLEX, 0.5, color.white)
    image[0:100, 100:200] = red

    green = np.zeros(shape, np.uint8) + color.green
    cv2.putText(green, 'Green', (25, 55), cv2.FONT_HERSHEY_COMPLEX, 0.5, color.white)
    image[0:100, 200:300] = green

    blue = np.zeros(shape, np.uint8) + color.blue
    cv2.putText(blue, 'Blue', (30, 55), cv2.FONT_HERSHEY_COMPLEX, 0.5, color.white)
    image[0:100, 300:400] = blue

    changeStylus = np.zeros(shape, np.uint8) + color.purple
    cv2.putText(changeStylus, 'Change', (20, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, color.white)
    cv2.putText(changeStylus, 'Stylus', (25, 65), cv2.FONT_HERSHEY_COMPLEX, 0.5, color.white)
    cv2.putText(changeStylus, 'Press "c" to change', (2, 80), cv2.FONT_HERSHEY_COMPLEX, 0.28, color.white)
    image[0:100, 400:500] = changeStylus

    return


def getOption(point):
    x, y = point

    if y > 100:
        return 4
    else:
        if 0 < x <= 100:
            return 0
        elif 100 < x <= 200:
            return 1
        elif 200 < x <= 300:
            return 2
        elif 300 < x <= 400:
            return 3
        elif 400 < x <= 500:
            return 5
        else:
            return 4


def loadROI():
    global lowers, uppers

    roi = np.load('roi.npy')

    total = [0, 0, 0]
    for i in roi:
        for j in i:
            total += j

    total = total / (roi.shape[0] * roi.shape[1])
    total = np.array(total, np.uint8)
    total = np.expand_dims(np.expand_dims(total, 0), 0)
    total = cv2.cvtColor(total, cv2.COLOR_BGR2HSV)

    up = lo = int(total[0][0][0])
    lo = 0 if lo < 10 else (lo - 10)
    up = 255 if up > 245 else (up + 10)

    lowers = np.array([lo, 100, 100], np.uint8)
    uppers = np.array([up, 255, 255], np.uint8)

    return


def changeStylus(vid):
    global color

    cv2.destroyAllWindows()

    roi = None

    while (True):
        ret, frame = vid.read()

        if not ret:
            break

        image = cv2.flip(frame, 1)
        st = 'Put stylus color inside rectangle and press "s" to save it and "d" to discard it'
        cv2.putText(image, st, (20, 80), cv2.FONT_HERSHEY_COMPLEX, 0.9, color.white)
        cv2.rectangle(image, (500, 300), (525, 325), color.green, 1)

        cv2.imshow('Image', image)

        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('d'):
            roi = None
            break
        elif key == ord('s'):
            roi = image[301:325, 501:525]
            # cv2.imshow('ROI', roi)
    cv2.destroyAllWindows()

    if roi is not None:
        np.save('roi.npy', roi)
        loadROI()
    return


def main():
    global color, points, lowers, uppers

    loadROI()

    colorId = 1  # green
    paintWindow = None

    vid = cv2.VideoCapture(0)

    vid.set(3, 1920)
    vid.set(4, 1080)

    # [106, 247, 125]
    while (True):
        ret, frame = vid.read()

        if not ret:
            break

        image = cv2.flip(frame, 1)
        frame = cv2.flip(frame, 1)

        drawOptions(image)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        extract = cv2.inRange(hsv, lowerb=lowers, upperb=uppers)

        if paintWindow is None:
            paintWindow = np.zeros(image.shape, np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        # extract = cv2.morphologyEx(extract, cv2.MORPH_OPEN, kernel, iterations=3)
        extract = cv2.erode(extract, kernel, iterations=2)
        extractk = cv2.morphologyEx(extract, cv2.MORPH_OPEN, kernel)
        extract = cv2.dilate(extract, kernel, iterations=1)

        contours, _ = cv2.findContours(extract, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            maxContour = max(contours, key=cv2.contourArea)
            mask = np.empty(image.shape, np.uint8)
            mask = np.full_like(mask, 255, np.uint8)
            center, radius = cv2.minEnclosingCircle(maxContour)
            cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), color[colorId], 1)
            # cv2.imshow('Mask', mask)

            option = getOption(center)

            if option == 0:
                paintWindow = np.zeros(image.shape, np.uint8)
                points[0] = deque([deque(maxlen=1024)])
                points[1] = deque([deque(maxlen=1024)])
                points[2] = deque([deque(maxlen=1024)])
            elif option == 1:
                colorId = 0
            elif option == 2:
                colorId = 1
            elif option == 3:
                colorId = 2
            elif option == 5:
                changeStylus(vid)

            if radius > 1:
                points[colorId][0].appendleft(center)
        else:
            if len(points[colorId][0]) > 0:
                points[colorId].appendleft(deque(maxlen=1024))

        for id in [0, 1, 2]:
            for point in points[id]:
                for i in range(len(point) - 1, 1, -1):
                    pt1 = point[i]
                    pt2 = point[i - 1]
                    cv2.line(paintWindow, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color[id], 8,
                             cv2.LINE_AA)
        image = cv2.addWeighted(image, 0.8, paintWindow, 0.8, 0)

        cv2.imshow('Image', image)
        # cv2.imshow('Extract', extract)

        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('c'):
            changeStylus(vid)

    vid.release()
    cv2.destroyAllWindows()


main()
