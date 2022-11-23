import cv2
import numpy as np
import face_recognition
import os

imgMom = face_recognition.load_image_file('images/MOM.jpeg')
imgMom = cv2.cvtColor(imgMom , cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file("images/MOMTEST.jpeg'")
imgTest = cv2.cvtColor(imgTest , cv2.COLOR_BGR2RGB)

#stores coordinates of top, right, bottom, left
faceLoc = face_recognition.face_locations(imgMom)[0]
encodeMom = face_recognition.face_encodings(imgMom)[0]
cv2.rectangle(imgMom ,(faceLoc[3] , faceLoc[0]) , ( faceLoc[1] , faceLoc[2]) , (255,0,255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest ,(faceLocTest[3] , faceLocTest[0]) , ( faceLocTest[1] , faceLocTest[2]) , (255,0,255), 2)

results= face_recognition.compare_faces([encodeMom], encodeTest)
faceDis = face_recognition.face_distance([encodeMom], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255) , 2)
cv2.imshow('Seema Kaushik' , imgMom)
cv2.imshow('Seema kaushik' , imgTest)

cv2.waitKey(0)