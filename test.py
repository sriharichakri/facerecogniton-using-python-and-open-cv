import tkinter as tk
from tkinter import Message, Text
import tkinter.messagebox as st
import tkinter.simpledialog as sd
import cv2
import csv
import os
import numpy as np
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font








def faceDetection(testimg):
    grayimg=cv2.cvtColor(testimg,cv2.COLOR_BGR2GRAY)
    classi=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces=classi.detectMultiScale(grayimg,scaleFactor=1.2,minNeighbors=5)

    return faces,grayimg


def labelsfortrainingdata(directory):
    faces=[]
    faceID=[]

    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system file")
                continue
            
           
            imgpath=os.path.join(path,filename)
            id=int(os.path.split(imgpath)[-1].split(".")[1])
            
            testimg=cv2.imread(imgpath)
            if testimg is None:
                print("Image not loaded properly")
                continue
            facesrect,grayimg=faceDetection(testimg)
            if len(facesrect)!=1:
               continue 
            (x,y,w,h)=facesrect[0]
            roigray=grayimg[y:y+w,x:x+h]
            faces.append(roigray)
            faceID.append(int(id))
    return faces,faceID
def traindata():
 facerecognizer = cv2.face.LBPHFaceRecognizer_create()
 faces,faceID=labelsfortrainingdata('dataset')
 facerecognizer=trainclassifier(faces,faceID)
 facerecognizer.write('trainingData.yml')
 res = "DAta trained succesfully "
 tk.messagebox.showinfo("alert",str(res))
 facerecognizer.read('trainingData.yml')


def trainclassifier(faces,faceID):
    facerecognizer=cv2.face.LBPHFaceRecognizer_create()
    facerecognizer.train(faces,np.array(faceID))
    return facerecognizer


def drawrect(testimg,face):
    (x,y,w,h)=face
    cv2.rectangle(testimg,(x,y),(x+w,y+h),(0,255,0),thickness=5)


def put_text(testimg,text,x,y):
    cv2.putText(testimg,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,2,(255,0,0),4)







def assurepathexists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
def takeimg():
 faceid=sd.askstring("id","enter your id")
 Name=sd.askstring("name","enter your name")
 vidcam = cv2.VideoCapture(0)


 facedetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


 count = 0

 assurepathexists("dataset/")


 while(True):

   
     _, imageframe = vidcam.read()

    
     gray = cv2.cvtColor(imageframe, cv2.COLOR_BGR2GRAY)

   
     faces = facedetector.detectMultiScale(gray, 1.3, 5)

 
     for (x,y,w,h) in faces:

        
         cv2.rectangle(imageframe, (x,y), (x+w,y+h), (255,0,0), 2)
        
        
         count += 1
        
       
         cv2.imwrite( "dataset/employ" + str(count) + '.' + str(faceid) + ".jpg", gray[y:y+h,x:x+w])

         cv2.imshow('frame', imageframe)

     
     if cv2.waitKey(100) & 0xFF == ord('q'):
         break

 
     elif count>=30:
          res = "Images Saved : " + faceid + " Name : " + Name
          tk.messagebox.showinfo("alert",str(res))
          
          break
     
 vidcam.release()
 ts = time.time()
 Date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
 Time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
 row = [faceid, Name, Date, Time]
 with open('employDetails.csv', 'a+') as csvFile:
      writer = csv.writer(csvFile, delimiter=',')
      writer.writerow(row)
      csvFile.close()



 cv2.destroyAllWindows()


def func():
 
 facerecognizer = cv2.face.LBPHFaceRecognizer_create()

 facerecognizer.read('trainingData.yml')

 cap=cv2.VideoCapture(0)

 while True:
     ret,testimg=cap.read()
     facesdetected,grayimg=faceDetection(testimg)



     for (x,y,w,h) in facesdetected:
       cv2.rectangle(testimg,(x,y),(x+w,y+h),(255,0,0),thickness=3)

     resizedimg = cv2.resize(testimg, (1000, 700))
     #cv2.imshow('face detection  ',resizedimg)
     #cv2.waitKey(10)


     for face in facesdetected:
         (x,y,w,h)=face
         roigray=grayimg[y:y+w, x:x+h]
         label,confidence=facerecognizer.predict(roigray)
         print("confidence:",confidence)
         print("label:",label)
         drawrect(testimg,face)
         
        
         if confidence > 70:
             
            cv2.rectangle(testimg,(x,y),(x+w,y+h),(0,255,0),thickness=3)
            put_text(testimg,str(label),x,y)
         else:
             id='unknown'
             tt=str(id)
             cv2.rectangle(testimg,(x,y),(x+w,y+h),(255,0,0),thickness=3)
             put_text(testimg,str(tt),x,y)
             

     resizedimg = cv2.resize(testimg, (1000, 700))
     cv2.imshow('face recognition  ',testimg)
     
     if cv2.waitKey(10) & 0xFF == ord('q'):
         break


 cap.release()
 cv2.destroyAllWindows




def login():
 window = tk.Tk()
 window.title("Face Recogniser")

 window.geometry('1280x720')
 window.configure(background='snow')
 window.grid_rowconfigure(0, weight=1)
 window.grid_columnconfigure(0, weight=1)

 message = tk.Label(window, text="Face Recognition System using python and opencv", bg="SpringGreen3", fg="white", width=50,
                   height=3, font=('times', 30, 'italic bold '))

 message.place(x=80, y=20)

 Notification = tk.Label(window, text="All things good", bg="Green", fg="white", width=15,
                      height=3, font=('times', 17, 'bold'))


 takeImg = tk.Button(window, text="Take Images",command=takeimg,fg="white"  ,bg="blue2"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
 takeImg.place(x=90, y=500)

 trainImg = tk.Button(window, text="Train Images",command=traindata,fg="white",bg="purple2"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
 trainImg.place(x=390, y=500)

 FA = tk.Button(window, text="Face recognition ",command=func,fg="white",bg="deep pink"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
 FA.place(x=690, y=500)

 quitWindow = tk.Button(window, text="Quit", command=window.destroy ,fg="white"  ,bg="Red"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
 quitWindow.place(x=990, y=500)

 window.mainloop()


def loginverify(sc):
    user=n2.get()
    pwd=p2.get()
    
    if(user=="asl"):
        if(pwd=="1234"):
            sc.destroy()
            login()
         
        else:
         
         st.showinfo("warning","wrong password")
    else:
     st.showinfo("warning","wrong username")



    
    



global sc
sc=tk.Tk()

sc.title("Login")

sc.geometry('640x320')

sc.configure(background='snow')


message1 = tk.Label(sc, text="Face Recognition System Login", bg="SpringGreen3", fg="white", width=30,
                   height=1, font=('times', 14, 'italic bold '))
 
message1.place(x=120, y=20)
n=tk.Label(sc, text="username", width=17, fg="white", bg="blue2", height=2, font=('times', 15, ' bold '))
n.place(x=80, y=90)


n2 = tk.Entry(sc, width=15, bg="white", fg="black", font=('times', 25, ' bold '))
n2.place(x=300, y=90)

p=tk.Label(sc, text="password", width=17, fg="white", bg="deep pink", height=2, font=('times', 15, ' bold '))
p.place(x=80, y=140)
p2 = tk.Entry(sc, width=15, bg="white", fg="white", font=('times', 25, ' bold '))
p2.place(x=300, y=140)

l= tk.Button(sc, text="Login",fg="white"  ,command=lambda:loginverify(sc),bg="SpringGreen3"  ,width=40  ,height=1 ,font=('times', 15, ' bold '))
l.place(x=80, y=200)  
n2.delete(first=0,last=10)  




