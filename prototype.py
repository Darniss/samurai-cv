import cv2
import face_recognition
import os 
import numpy as np
from datetime import datetime
import pickle
from numba import jit, cuda
import numba
import re
import pyttsx3

path = 'staffs'
images = []
classNames = []

mylist = os.listdir(path)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print(len(mylist))



def findEncodings(images):
    encodeList = []
    counter=0
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
         counter+=1
         encoded_face = face_recognition.face_encodings(img)[0]
         print(counter)
        except IndexError as e:
         print(e)
        encodeList.append(encoded_face)
        
    return encodeList

# print(images)

encoded_face_train = findEncodings(images)

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'\n{name}, {time}, {date}')

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
print (volume)                          #printing current volume level
engine.setProperty('volume',1.0)    # setting up volume level  between 0 and 1

def speak(audio):
    engine.say(audio) 
    engine.runAndWait()

# take pictures from webcam 
cap  = cv2.VideoCapture(0)
window_name='Samurai'

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)

    for encode_face, faceloc in zip(encoded_faces,faces_in_frame):
        matches = face_recognition.compare_faces(encoded_face_train, encode_face)
        faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
        matchIndex = np.argmin(faceDist)
        print(matchIndex)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper().lower()
            sep = ' ' #\s+
            stripped = name.split(sep, 1)[0]
            _name1= ''.join(e for e in stripped if e.isalnum())#''.join(i for i in name if not i.isdigit())
            _name=''.join(i for i in _name1 if not i.isdigit())
            print(_name)
            y1,x2,y2,x1 = faceloc
            # since we scaled down by 4 times
            y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img,_name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)          
            markAttendance(name)
            speak("Hello "+_name)
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  . venv\scripts\activate