import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
from tkinter import PhotoImage
from PIL import Image, ImageTk
import tkinter.simpledialog as tsd
import cv2,os
import csv
import numpy as np
import datetime
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


######################################## CSV HELPER (NO PANDAS) ########################################

def get_student_details(serial):
    try:
        with open("StudentDetails\\StudentDetails.csv", newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    if str(serial) == row[0]:
                        return row[2], row[4]
                except:
                    continue
    except:
        pass
    return None, "Unknown"


######################################## BASIC FUNCTIONS ########################################

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def tick():
    current_time = time.strftime('%I:%M:%S %p')
    clock.config(text=current_time)
    clock.after(1000, tick)

def contact():
    mess._show(title='Contact us', message="Please contact us on : 'shubhamkumar8180323@gmail.com' ")

def check_haarcascadefile():
    if not os.path.isfile("haarcascade_frontalface_default.xml"):
        mess._show(title='Missing File', message='haarcascade_frontalface_default.xml missing')
        window.destroy()


######################################## IMAGE CAPTURE ########################################

def TakeImages():
    check_haarcascadefile()

    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")

    serial = 0
    exists = os.path.isfile("StudentDetails\\StudentDetails.csv")

    if exists:
        with open("StudentDetails\\StudentDetails.csv", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            for l in reader1:
                serial += 1
        serial = (serial // 2)
    else:
        with open("StudentDetails\\StudentDetails.csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(['SERIAL NO.','','ID','','NAME'])
            serial = 1

    Id = txt.get()
    name = txt2.get()

    if ((name.isalpha()) or (' ' in name)):

        # CAMERA FIX HERE
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cam.isOpened():
            mess._show(title='Camera Error', message='Unable to access camera')
            return

        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        sampleNum = 0

        while True:
            ret, img = cam.read()
            if not ret:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # LAG REDUCTION
            faces = detector.detectMultiScale(gray, 1.2, 3)

            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                sampleNum += 1

                cv2.imwrite(
                    "TrainingImage\\"+name+"."+str(serial)+"."+Id+"."+str(sampleNum)+".jpg",
                    gray[y:y+h,x:x+w]
                )

                cv2.imshow('Taking Images', img)

            if cv2.waitKey(1)==ord('q') or sampleNum>100:
                break

        cam.release()
        cv2.destroyAllWindows()

        row = [serial,'',Id,'',name]
        with open('StudentDetails\\StudentDetails.csv','a+') as csvFile:
            writer=csv.writer(csvFile)
            writer.writerow(row)

        message1.configure(text="Images Taken for ID : "+Id)
    else:
        message.configure(text="Enter Correct Name")


######################################## TRAIN MODEL ########################################

def TrainImages():
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel/")

    # FIXED RECOGNIZER
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces,ID = getImagesAndLabels("TrainingImage")

    try:
        recognizer.train(faces,np.array(ID))
    except:
        mess._show(title='No Registrations', message='Register someone first!')
        return

    recognizer.save("TrainingImageLabel\\Trainner.yml")
    message1.configure(text="Profile Saved Successfully")


def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    Ids=[]

    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        ID=int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(ID)

    return faces,Ids


######################################## ATTENDANCE ########################################

def TrackImages():
    check_haarcascadefile()

    assure_path_exists("Attendance/")
    assure_path_exists("StudentDetails/")

    recognizer=cv2.face.LBPHFaceRecognizer_create()

    if not os.path.isfile("TrainingImageLabel\\Trainner.yml"):
        mess._show(title='Data Missing', message='Please Save Profile first!')
        return

    recognizer.read("TrainingImageLabel\\Trainner.yml")

    faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # CAMERA FIX
    cam=cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        mess._show(title='Camera Error', message='Unable to access camera')
        return

    font=cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, im = cam.read()
        if not ret:
            break

        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

        # LAG REDUCTION
        faces=faceCascade.detectMultiScale(gray,1.2,3)

        for(x,y,w,h) in faces:
            serial,conf=recognizer.predict(gray[y:y+h,x:x+w])

            if conf<50:
                ID,bb=get_student_details(serial)

                ts=time.time()
                date=datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp=datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

                attendance=[str(ID),'',bb,'',str(date),'',str(timeStamp)]
            else:
                bb="Unknown"

            cv2.putText(im,str(bb),(x,y+h),font,1,(255,255,255),2)

        cv2.imshow('Taking Attendance',im)

        if cv2.waitKey(1)==ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


######################################## GUI (UNCHANGED) ########################################

window = tk.Tk()
window.geometry("1280x720")
window.resizable(True,False)
window.title("Attendance System")
window.configure(background='#2d420a')

frame1 = tk.Frame(window, bg="#d3c9d1")
frame1.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.80)

frame2 = tk.Frame(window, bg="#d3c9d1")
frame2.place(relx=0.51, rely=0.17, relwidth=0.38, relheight=0.80)

message3 = tk.Label(window,text="Face Recognition Based Attendance Monitoring System",
fg="white",bg="#2d420a",width=55,height=1,font=('comic',29,'bold'))
message3.place(x=10,y=10)

clock = tk.Label(window,fg="#ff61e5",bg="green",width=20,font=('comic',15,'bold'))
clock.place(relx=0.50,rely=0.09)
tick()

lbl=tk.Label(frame2,text="Enter ID",bg="#c79cff",font=('comic',17,'bold'))
lbl.place(x=80,y=55)

txt=tk.Entry(frame2,width=32,font=('comic',15,'bold'))
txt.place(x=30,y=88)

lbl2=tk.Label(frame2,text="Enter Name",bg="#c79cff",font=('comic',17,'bold'))
lbl2.place(x=80,y=140)

txt2=tk.Entry(frame2,width=32,font=('comic',15,'bold'))
txt2.place(x=30,y=173)

message1=tk.Label(frame2,text="",bg="#c79cff",width=39,font=('comic',15,'bold'))
message1.place(x=7,y=230)

message=tk.Label(frame2,text="",bg="#c79cff",width=39,font=('comic',16,'bold'))
message.place(x=7,y=450)

takeImg=tk.Button(frame2,text="Take Images",command=TakeImages,
bg="#6d00fc",fg="white",width=34,font=('comic',15,'bold'))
takeImg.place(x=30,y=300)

trainImg=tk.Button(frame2,text="Save Profile",command=TrainImages,
bg="#6d00fc",fg="white",width=34,font=('comic',15,'bold'))
trainImg.place(x=30,y=380)

trackImg=tk.Button(frame1,text="Take Attendance",command=TrackImages,
bg="#3ffc00",width=13,font=('comic',12,'bold'))
trackImg.place(x=160,y=85)

window.mainloop()