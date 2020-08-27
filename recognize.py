'''
Peter Smiley

TO EXIT: Press esc button

Draws black rectangle around face
Draws white rectangles around eyes
Draws rectangle around color blobs with colors corresponding to blob color

Helpful post for choosing color ranges in color blob detection:
https://stackoverflow.com/questions/48528754/what-are-recommended-color-spaces-for-detecting-orange-color-in-open-cv
'''

from time import sleep
import numpy as np
import cv2

# camera control
key = cv2. waitKey(1)
camera = cv2.VideoCapture(0)

# face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# color detection based on hsv scale
boundaries = [
    # dark red:
	([165, 130, 100], # x start
    [180, 255, 255], # x end
    (0, 0, 255)), # bgr color used to draw box

    # light red:
    ([0, 130, 100], 
    [10, 255, 255],
    (0, 0, 255)), 

    # orange:
    ([10, 130, 100], 
    [25, 255, 255],
    (0, 140, 255)), 

    # yellow:
    ([0, 130, 100], 
    [20, 255, 255],
    (40, 255, 255)), 

    # green:
    ([40, 130, 100], 
    [90, 255, 255],
    (0, 255, 0)), 
  
    # blue:
    ([90, 130, 100], 
    [130, 255, 255],
    (255, 0, 0)), 

    # purple / pink:
	([135, 130, 100],
    [165, 255, 255], 
    (255, 0, 255)) 
]


img_count = 0 


while True:

    try:
        # capture with webcam
        check, frame = camera.read()
        # cv2.imshow("Capturing", frame)
        if not check:
            break
        key = cv2.waitKey(1)

    
        # take and save image
        cv2.imwrite(filename='saved_img.jpg', img=frame)
        img = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
        
        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            # draws black rectangle around face
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]  
            eyes = eye_cascade.detectMultiScale(roi_gray)
            # draws white rectangle around eyes
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,255),2)
        
        hsv = cv2.medianBlur(img.copy(), 3) # eliminates a lot of noise, raise 2nd param to pick up less 
        hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV) # converts to hsv color system help detecting color intensity
        # loop through different possible colors
        for (lower, upper, rect_color) in boundaries:
            lower = np.array(lower, dtype = "uint8")
            upper = np.array(upper, dtype = "uint8")

            # converts pixels to white if their color is in the bounds. Else -> black.
            frame_threshed = cv2.inRange(hsv, lower, upper)

            # test color pickups 
            # cv2.imshow(str(rect_color), frame_threshed)
            # cv2.waitKey(0)
            
            ret,thresh = cv2.threshold(frame_threshed,127,255,0)

            # creates contors around white pixel clusters
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(c) for c in contours] # gets list of all contors

            max_index = 0
            cnt = []
            if len(areas) != 0:
                max_index = np.argmax(areas) 
                # gets the biggest contor from the list of contors
                cnt = contours[max_index] 
            
            if (len(cnt) > 300): # only picks up blobs with area greater than 300
                x,y,w,h = cv2.boundingRect(cnt) # creates rectangle to encompass largest contor
                cv2.rectangle(img,(x,y),(x+w,y+h),rect_color,2) # draws rectangle over image in color corresponding to blob color

        # # save image in local directory 
        # img = cv2.imwrite(filename='frame_{0}.jpg'.format(img_count), img=img)
        # print("Image # {0} saved!".format(img_count))
        # img_count += 1

        cv2.imshow("marked_img", img)

                                
        if key == 27: # 27 ascii is escape
            camera.release()
            cv2.destroyAllWindows()
            break
    
    except(KeyboardInterrupt):
        camera.release()
        print("Camera off.")
        cv2.destroyAllWindows()
        break