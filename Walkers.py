import cv2


# Create our body classifier
body_classifier=cv2.CascadeClassifier('C:/Users/mitta_ck4oqhq/Downloads/PRO-106-ProjectTemplate-main/PRO-106-ProjectTemplate-main/haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('C://Users/mitta_ck4oqhq/Downloads/PRO-106-ProjectTemplate-main/PRO-106-ProjectTemplate-main/walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale

    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Pass frame to our body classifier
    bodies=body_classifier.detectMultiScale(grey,1.1,5)
    
    # Extract bounding boxes for any bodies identified
    for(x,y,w,h)in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(105,255,255),2)
        body=frame[y:y+h,x:x+w]
        cv2.imshow('frame',frame)

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
