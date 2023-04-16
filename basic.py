import numpy as np
import cv2
import face_recognition

imgpawan=face_recognition.load_image_file('images/Pawan.jpg')
imgpawan=cv2.cvtColor(imgpawan,cv2.COLOR_BGR2RGB)

imgpawantest=face_recognition.load_image_file('images/karan.jpg')
imgpawantest=cv2.cvtColor(imgpawantest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgpawan)[0]
encodepawan=face_recognition.face_encodings(imgpawan)[0]
cv2.rectangle(imgpawan,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLoctest = face_recognition.face_locations(imgpawantest)[0]
encodepawantest=face_recognition.face_encodings(imgpawantest)[0]
cv2.rectangle(imgpawantest,(faceLoctest[3],faceLoctest[0]),(faceLoctest[1],faceLoctest[2]),(255,0,255),2)

results=face_recognition.compare_faces([encodepawan],encodepawantest)
faceDis=face_recognition.face_distance([encodepawan],encodepawantest)
print(faceDis)
if(faceDis<0.5):
    cv2.putText(imgpawantest,f'Same Person {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
else:
    cv2.putText(imgpawantest, f'Different Person {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),2)

cv2.imshow("Pawan Kumar",imgpawan)
cv2.imshow("Pawan Kumar Test Image",imgpawantest)
cv2.waitKey(0)