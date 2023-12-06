from keras.models import load_model
from time import sleep
from keras_preprocessing.image import load_img, img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import random

face_classifier = cv2.CascadeClassifier(r'D:\emotion detector\Emotion_Detection_CNN\haarcascade_frontalface_default.xml')
classifier =load_model(r'D:\emotion detector\Emotion_Detection_CNN\model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
Happy=["The only joy in the world is to begin.","It is only possible to live happily ever after on a daily basis","Remember this, that very little is needed to make a happy life"]
Sad=["It’s amazing how someone can break your heart and you can still love them with all the little pieces."]
Neutral=["show some emotion "]
Surprise=["Surprise is the greatest gift which life can grant us."]
Angry=["Angry people are not always wise."]
Fear=["Do one thing every day that scares you"]
Disgust=["We are so accustomed to disguise ourselves to others, that in the end, we become disguised to ourselves."]

cap = cv2.VideoCapture(0)

root = tk.Tk()
root.geometry("700x700");
root.title('Emotion Detector')
root.config(bg='#4fe3a5')
def close():
    root.destroy()
def capture():
    global cap,label
    print("Capturing image...")
    ret, frame = cap.read()
    if ret:
        blurred = cv2.GaussianBlur(frame, (15, 15), 0)
        
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            cv2.rectangle(blurred,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = classifier.predict(roi)[0]
                label=emotion_labels[prediction.argmax()]
                label_position = (x,y)
                if label=="Neutral":
                    text=random.choice(Neutral)
                elif label=="Happy":        
                    text=random.choice(Happy)
                elif label=="Sad":
                    text=random.choice(Sad)
                elif label=="Surprise":
                    text=random.choice(Surprise)
                elif label=="Angry":
                    text=random.choice(Angry)
                elif label =="Fear":
                    text=random.choice(Fear)
                elif label=="Disgust":
                    text=random.choice(Disgust)
            
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(blurred, text, (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                

        cv2.imwrite("captured_image11.jpg", blurred)
        print("Image captured and saved!")

    else:
        print("Error capturing image!")
    cap.release()
    
    root.after(10, update)

# Create a canvas widget to display the video feed
canvas = tk.Canvas(root, width=640, height=480)
canvas.place(x=100, y=100)
canvas.grid(row=4, column=4, padx=10, pady=20)
canvas.pack()
closebutton=tk.Button(root, text="Close", font=("Helvetica", 16),command=close)
closebutton.place(x=0, y=0, anchor="nw")
button=tk.Button(root, text="Capture", font=("Helvetica", 16),command=capture)
button.configure(bg="#4CAF50")
button.configure(fg="white")
button.configure(borderwidth=0, relief="flat")
button.configure(activebackground="#3E8E41", activeforeground="white")
button.place(relx=.5 , rely=.6,anchor= tk.CENTER)

def update():
    global cap
    if cap is None:  # If camera is closed, open it
        cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.imgtk = imgtk
        canvas.create_image(0,0, anchor=tk.NW, image=imgtk)
    root.after(10, update)
root.after(10, update)
root.attributes('-fullscreen', True)
# Run the Tkinter event loop
root.mainloop()

# Release the video capture object
cap.release()