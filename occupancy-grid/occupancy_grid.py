import cv2
import cv2.aruco as aruco
import numpy as np
import os

import serial
import serial.tools.list_ports as list_ports

PID_MICROBIT = 516
VID_MICROBIT = 3368
TIMEOUT = 0.1

def find_comport(pid, vid, baud):
    ''' return a serial port '''
    ser_port = serial.Serial(timeout=TIMEOUT)
    ser_port.baudrate = baud
    ports = list(list_ports.comports())
    print('scanning ports')
    for p in ports:
        print('port: {}'.format(p))
        try:
            print('pid: {} vid: {}'.format(p.pid, p.vid))
        except AttributeError:
            continue
        if (p.pid == pid) and (p.vid == vid):
            print('found target device pid: {} vid: {} port: {}'.format(
                p.pid, p.vid, p.device))
            ser_port.port = str(p.device)
            return ser_port
    return None

def findArucoMarkers(img, markerSize = 5, totalMarkers=250, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, _ = aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)
    if draw:
        aruco.drawDetectedMarkers(img, bboxs)
    return [bboxs, ids]

def arucoAug(bbox, distance):
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    forward = (int((tl[0] + tr[0]) / 2), int((tl[1] + tr[1]) / 2))
    position = (int((tl[0] + br[0])/2),int((tl[1] + br[1])/2))
    mag = ((position[1]-forward[1])**2 + (position[0]-forward[0])**2)**(1/2)
    vec = ((forward[0]-position[0])*distance/mag,(forward[1]-position[1])*distance/mag)
    forward = (int(position[0] + vec[0]),int(position[1] + vec[1]))
    return position, forward

def drawInit(img,count_x,count_y):
    gridRect = []
    score = []
    allocated = []
    x,y = 0,0
    width,height = img.shape[0]//count_x,img.shape[0]//count_y
    for _ in range(count_y):
        tmp = []
        stmp = []
        atmp = []
        for _ in range(count_x):
            cv2.rectangle(img, (x, y), (x+width, y+height), (0, 0, 0), -1)
            tmp.append(((x, y), (x+width, y+height)))
            stmp.append(0)
            atmp.append(False)
            x += width
        x = 0
        y += height
        gridRect.append(tmp)
        score.append(stmp)
        allocated.append(atmp)
    return gridRect,score,allocated

def drawGrid(img,count_x,count_y,score,allocated):
    width = img.shape[0]//count_x
    height = img.shape[1]//count_y
    for i in range(count_y):
        for j in range(count_x):
            if allocated[i][j]:
                greenscore = -score[i][j]/5 + 1/2
                redscore = score[i][j]/5 + 1/2
                if greenscore < 0:
                    greenscore = 0
                if redscore < 0:
                    redscore = 0
                cv2.rectangle(img, (j*width, i*height), (j*width+width, i*height+height), (0, greenscore, redscore), -1)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def checkIntersect(position,forward,gridRect,score,allocated):
    for i in range(len(gridRect)):
        for j in range(len(gridRect[i])):
            v0 = (gridRect[i][j][0][0],gridRect[i][j][0][1])
            v1 = (gridRect[i][j][1][0],gridRect[i][j][0][1])
            v2 = (gridRect[i][j][0][0],gridRect[i][j][1][1])
            v3 = (gridRect[i][j][1][0],gridRect[i][j][1][1])
            i0 = intersect(v0, v1, position, forward)
            i1 = intersect(v1, v3, position, forward)
            i2 = intersect(v3, v2, position, forward)
            i3 = intersect(v2, v0, position, forward)
            if gridRect[i][j][0][0]<=forward[0] and forward[0]<=gridRect[i][j][1][0] and gridRect[i][j][0][1]<=forward[1] and forward[1]<=gridRect[i][j][1][1]:
                if not allocated[i][j]:
                    allocated[i][j] = True
                if score[i][j] < 10:
                    score[i][j] += 2
            elif i0 or i1 or i2 or i3:
                if not allocated[i][j]:
                    allocated[i][j] = True
                if score[i][j] > -10:
                    score[i][j] -= 1

def main():

    print('looking for microbit')
    ser_micro = find_comport(PID_MICROBIT, VID_MICROBIT, 115200)
    if not ser_micro:
        print('microbit not found')
        return
    print('opening and monitoring microbit port')
    ser_micro.open()

    cap = cv2.VideoCapture(0)
    mapImg = np.zeros((600, 600, 4))

    gridRect,score,allocated = drawInit(mapImg,100,100)

    while True:
        line = ser_micro.readline().decode('utf-8')
        _, img = cap.read()
        if line:
            sensor = line.strip()
            print(sensor)
            sensor = int(sensor)
            print(sensor)
            arucofound = findArucoMarkers(img)
            drawGrid(mapImg, 100, 100,score,allocated)
            # # loop through all the markers and augment each one
            if len(arucofound[0])!=0:
                bbox = arucofound[0][0]
                pos, fwd = arucoAug(bbox, sensor)
                checkIntersect(pos, fwd, gridRect, score, allocated)
        cv2.imshow('img',img)
        cv2.imshow('map',mapImg)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    ser_micro.close()
main()
