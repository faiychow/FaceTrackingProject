import cv2

#Load pretrained model inside cv2 directory
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Start camera
cap = cv2.VideoCapture(0)

#indefinite loop
while True:
    #ret is if frame was successfully grabbed from the camera and frame is image data both being returned by cap.read()
    ret, frame = cap.read()
    if ret == False:
        print("Failed to grab frame")
        break

    #Flip to mirror
    #0 = vertical flip, 1 = horizontal flip, -1 = both flip
    frame = cv2.flip(frame, 1)

    #Convert to gray cuz face detection works better on grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #parameters explained: 
    #scaleFactor: 1.1 means the image size is reduced by 10%. Smaller scaleFactor = better detection, increased processing time.
    #minNeighbors: How many boxes allowed next to each other, Higher = fewer detections, higher confidence. Low = incorrect detections
    #minSize: minimum size to be detected, by pixels

    #x, y, w, h are returned by .detectMultiScale (x,y) upper left coordinate of detected face, w/h is width/height
    #So for every full reurn of values or coordinates, use .rectangle function to make rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        #Calculate center of the rectangle
        center_x = x + w // 2
        center_y = y + h // 2
        
        #draw a line on the frame from (center_x - 10, y) to (center_x + 10, y) to make a line and so on
        #last parameter = line thickness
        cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 255, 0), 2)  # Horizontal line
        cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 255, 0), 2)  # Vertical line

    #Display result, window name is first parameter
    cv2.imshow('Face Detection', frame)

    #breaks outa loop when q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#releases capture and detroys everything after process
cap.release()
cv2.destroyAllWindows()
