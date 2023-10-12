import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
import os

#Load the Face Recognition YOLO model
face_model = YOLO("models/yolov8n-face.pt")
# face_model.fuse()
#Load the License Plate Recognition YOLO model
license_plate_model = YOLO("models/license_plate_detector.pt")
#license_plate_model.fuse()

#Load our sample video
sample_video = cv2.VideoCapture("sample/videos/sample.mp4")

#Load our webcam
cam = cv2.VideoCapture(0)

#Test if cam failed to open
while not cam.isOpened():
    print("Cam is not open!")

count = 0

#If cam successfully opened, run the program
while cam.isOpened():
    
    #Read if frame retrieval is succesful, and read the retrieved frame
    success, image = cam.read()

    #If frame retrieval succesful, process the frame and perform inference on the frame
    if success:
        #Perform YOLOv8 inference
        results = face_model.predict(image)
        
        person_found = False

        
        #Plot the inference results, to be displayed in a window
        annotated_frame = results[0].plot()

        #Retrieve properties about inferred result, check if it detects a face (face class stored in cls variable as 0)
        for box in results[0].boxes.cpu().numpy():
            cls = int(box.cls[0])
            if cls == 0:
                person_found = True
        
        #If a person's face detected, save the annotated result into a JPG file in the results directory.
        if person_found:
             if count % 5 == 0:
                 t = time.localtime()
                 current_time = time.strftime("%H-%M-%S", t)
                 filename = "person_" + current_time + ".jpg"
                 absolute_path = os.path.join(os.getcwd(), 'results', filename)
                 print(absolute_path)
                 cv2.imwrite(absolute_path, annotated_frame)

        #Show the video frame in screen for demonstration purposes
        cv2.imshow("Real Time Detection, Press Q to Quit", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        count += 1


cam.release()
cv2.destroyAllWindows()

