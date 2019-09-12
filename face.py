from __future__ import print_function
import cv2 as cv
import base64
import argparse

def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    faces = face_cascade.detectMultiScale(frame_gray)

    # DESENHAR CIRCULOS EM VOLTA DAS FACES ENCONTRADAS
    #
    # for (x,y,w,h) in faces:
    #     center = (x + w//2, y + h//2)
    #
    #     # frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
    #     faceROI = frame_gray[y:y+h,x:x+w]
    #
        # DETECTAR E DESENHAR CIRCULOS NOS OLHOS DE CADA FACE
        #
        # eyes = eyes_cascade.detectMultiScale(faceROI)
        # for (x2,y2,w2,h2) in eyes:
        #     eye_center = (x + x2 + w2//2, y + y2 + h2//2)
        #     radius = int(round((w2 + h2)*0.25))
        #     frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
    

    # MOSTRAR JANELA COM IMAGEM DA CAMERA
    #
    # cv.imshow('Capture - Face detection', frame)
    #
    #

    if len(faces) > 0:
        retval, buf	= cv.imencode('.jpg', frame)
        b64 = base64.b64encode(buf.tostring())
        return b64.decode('utf-8')
    else:
        return None


    

parser = argparse.ArgumentParser(
    description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.',
                    default=cv.data.haarcascades + 'haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.',
                    default=cv.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument(
    '--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
# -- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
camera_device = args.camera
# -- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break

    b64_img = detectAndDisplay(frame)

    if b64_img is not None:
        print(b64_img)

    if cv.waitKey(10) == 27:
        break
