import numpy as np
import cv2
import pickle
import smtplib
import pandas as pd
#haarcascade


def sendEmail(to , content):
    server = smtplib.SMTP('smtp.gmail.com',587)
    server.ehlo()
    server.starttls()
    server.login('Email_id','Password') #go to gmail and change secuirty settings to allow third party app
    server.sendmail('Email_id',to,content)
    server.close()

name = []
checker =[]
face_cascade =cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')

labels = {"person_name":1}
with open("labels.open",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while(True):

    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces =face_cascade.detectMultiScale(gray, scaleFactor=1.5 ,minNeighbors=5)
    for(x , y , w, h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h,x:x+w] #ycord1_start ,ycord_end
        roi_color = frame[y:y+h,x:x+w]

        #Recognize

        id_,  conf = recognizer.predict(roi_gray)
        if conf>=75:
            #print( id_ ) 
            #print(labels[id_])
            font =cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame,name+"No Mask",(x,y),font,1,color,stroke,cv2.LINE_AA)
            if labels[id_] in name:
                pass
            else:
                name.append(labels[id_])
        

        
                    
        data = pd.read_csv("Data.csv")

        row1=str(data.iloc[:,0])
        col=str(data.iloc[:,2])
        row1=row1.split()
        col=col.split()   

        
        value =row1.index(name)
        mail_id=col[value]
        print(col[value])
        
        

        if len(checker)<=3:
            if name in checker:
                pass
            else:
                checker.append(name)
                content="Please wear your mask properly"
                to=mail_id
                sendEmail(to,content)       
                print(checker)
        else:
            checker.clear()
    
           




        img_item = "7.png"
        cv2.imwrite(img_item, roi_color)

        
        color = (255, 0 , 0) #BGR not rgb
        stroke=2
        end_cord_x= x+w
        end_cord_y= y +h
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)

  

    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break




cap.release()
cv2.destroyAllWindows()