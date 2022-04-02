import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw
import random
import cvzone


mouth_cascade = cv.CascadeClassifier('haarcascade_mcs_mouth.xml')
faceDetector  = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeDetector   = cv.CascadeClassifier("haarcascade_eye.xml")

stickerMouth = cv.imread("images/1.png"  ,cv.IMREAD_UNCHANGED)
sticker      = cv.imread("images/2.jpg" ,cv.IMREAD_UNCHANGED)
stickerEye   = cv.imread("images/result.png"    ,cv.IMREAD_UNCHANGED)



# if you want to use the sticker on your eyes or your lips you must to do 
'''
img=Image.open("images/1.jpg").convert("RGB")
npImage=np.array(img)
h,w=img.size

alpha = Image.new('L', img.size,0)
draw = ImageDraw.Draw(alpha)
draw.pieslice([0,0,h,w],0,360,fill=255)

# Convert alpha Image to numpy array
npAlpha=np.array(alpha)

# Add alpha layer to RGB
npImage=np.dstack((npImage,npAlpha))

# Save with alpha
a = Image.fromarray(npImage).save('images/result.png')
'''

videoCap = cv.VideoCapture(-1)
ds_factor = 1
font = cv.FONT_ITALIC
fontSize = 0.3
textColor = (255, 255, 255)
thickness = 1

while True:    

    ret, frame = videoCap.read()
    if ret == False:
        break

    
    #cv.rectangle(frame, (0, 0), (275, 95), (0,0,0), -1)
    #cv.putText(frame, 'press key 1: Place a non-square sticker on the face', (10, 10), font, fontSize, textColor, thickness)
    #cv.putText(frame, 'press key 2: Place the sticker on the eyes and lips', (10, 30), font, fontSize, textColor, thickness)
    #cv.putText(frame, 'press key 3: Place a blur on the face',               (10, 50), font, fontSize, textColor, thickness)
    #cv.putText(frame, 'press key 4: edge detection on the face',             (10, 70), font, fontSize, textColor, thickness)
    #cv.putText(frame, 'press key 5: rotate video by 45 degrees',             (10, 90), font, fontSize, textColor, thickness)
    
    key = cv.waitKey(100)

    if ret:

        faces = faceDetector.detectMultiScale(frame,1.3)
        eyes  = eyeDetector.detectMultiScale(frame, 1.1, maxSize=(65 , 65))  
        mouth = mouth_cascade.detectMultiScale(frame,1.7, 11)
        
        
        if key == 49: # press key 1 ASCII

            for face in faces:
                x,y,w,h = face
                sticker_resized = cv.resize(sticker, (w, h))
                #overlay two images of a scene
                frame = cvzone.overlayPNG(frame, sticker_resized, [x, y])


        elif key == 50: # press key 2 ASCII
            
            for eye in eyes:
                for eye in eyes:
                    xe,ye,we,he = eye
                    #resize the glasses so that they fit Hermione perfectly
                    stickerEye=cv.resize(stickerEye,(we+5,he+5))
#we simply have to replace the pixels of the image of Hermione with the pixels of the stickerEye.
#To do that we use 2 for loops. If the pixel on the stickerEye image is 0 that means, 
#we want that portion to be transparent, so we do not replace that pixel
                    for i in range(stickerEye.shape[0]):
                        for j in range(stickerEye.shape[1]):
                            if (stickerEye[i,j,3]>0):
                                frame[ye+i-5,xe+j-8, :]=stickerEye[i,j,:-1]

            
            for lips in mouth:
                xl,yl,wl,hl = lips
                sticker_resized = cv.resize(stickerMouth, (wl, hl))
                frame = cvzone.overlayPNG(frame, sticker_resized, [xl, yl])
        
        elif key == 51: # press key 3 ASCII
            for (x,y,w,h) in faces:
                roi = frame[y:y+h, x:x+w]
                # applying a gaussian blur over this new rectangle area
                roi = cv.GaussianBlur(roi, (23, 23), 30)
                # impose this blurred image on original image to get final image
                frame[y:y+roi.shape[0], x:x+roi.shape[1]] = roi


        elif key == 52: # press key 4 ASCII
            # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # _, mask = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
            # using cv.Canny() for edge detection.
            frame = cv.Canny(frame, 100, 200)
        
        elif key == 53: # press key 5 ASCII
            # grab the dimensions of the image and calculate the center of the image
            (h, w) = frame.shape[:2]
            (cX, cY) = (w // 2, h // 2) 
            # rotate our image by 45 degrees around the center of the image
            M = cv.getRotationMatrix2D((cX, cY), 45, 1.0)
            frame = cv.warpAffine(frame, M, (w, h))
            



        
        cv.imshow("frame", frame)

