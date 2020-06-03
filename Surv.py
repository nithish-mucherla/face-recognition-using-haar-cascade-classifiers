import numpy as np
import cv2
from PIL import Image
import os

path = 'F:/howle/Desktop/COURSES/WIN SEM 19-20/MLt/J/Face recog/faces/'
cascadePath = 'F:/howle/Desktop/COURSES/WIN SEM 19-20/MLt/J/Face recog/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascadePath)
recognizer = cv2.face.LBPHFaceRecognizer_create()
def create_db():    
    cap = cv2.VideoCapture(0)
    id = input('enter user id')
    sampleN=0;

    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            sampleN=sampleN+1;
            cv2.imwrite("F:/howle/Desktop/COURSES/WIN SEM 19-20/MLt/J/Face recog/faces/."+str(id)+ "." +str(sampleN)+ ".jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.waitKey(100)
        cv2.imshow('img',img)
        cv2.waitKey(1)
        if sampleN > 200:
            break
    cap.release()
    cv2.destroyAllWindows()

def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = face_cascade.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

def train():
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    print(ids)
    recognizer.train(faces, np.array(ids))
    # Save the model into trainer/trainer.yml
    recognizer.write('F:/howle/Desktop/COURSES/WIN SEM 19-20/MLt/J/Face recog/trash.yml') # recognizer.save() worked on Mac, but not on Pi
    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
    
def face_recog():
    l=[]
    recognizer.read('F:/howle/Desktop/COURSES/WIN SEM 19-20/MLt/J/Face recog/trash.yml')
    faceCascade = cv2.CascadeClassifier(cascadePath);
    font = cv2.FONT_HERSHEY_SIMPLEX
    id = 0
    # names related to ids: example ==> Marcelo: id=1,  etc
    names = ['none','nithish','Laddu']
    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height
    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    while True:
        ret, img =cam.read()
        #img = cv2.flip(img, -1) # Flip vertically
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale( 
                gray, 
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(minW), int(minH)),
                )    
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (0 < confidence < 70):
                if(id not in l):
                    l.append(id)
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))        
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            #cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)      
        cv2.imshow('camera',img) 
        if(cv2.waitKey(100) == 27):
            break
        # Do a bit of cleanup
    cam.release()
    cv2.destroyAllWindows()
    return l