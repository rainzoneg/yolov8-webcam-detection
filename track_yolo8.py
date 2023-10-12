import cv2
from ultralytics import YOLO
import time
import os

WEBCAM_URL = "placeholder";

#Load the Face Recognition YOLO model & improve memory efficiency
face_model = YOLO("models/yolov8n-face.pt")
face_model.fuse()
#Load the License Plate Recognition YOLO model & improve memory efficiency
license_plate_model = YOLO("models/license_plate_detector.pt")
license_plate_model.fuse()

#Load our sample video
sample_video = cv2.VideoCapture("sample/videos/sample.mp4")

#Load our webcam
cam = cv2.VideoCapture("rtsp://192.168.1.5:8080/h264_ulaw.sdp")

#Test if cam failed to open
while not cam.isOpened():
    print("Cam is not open!")

#If cam successfully opened, run the program
while cam.isOpened():
    
    #Read if frame retrieval is succesful, and read the retrieved frame
    success, image = cam.read()

    #If frame retrieval succesful, process the frame and perform inference on the frame
    if success:
        #Perform YOLOv8 inference -> tracking mode, persist id/box between cam frames
        trackResults = face_model.track(image, persist=True)

        #Plot the inference results, to annotate with rect box, text
        annotated_frame = trackResults[0].plot()

        #Get the detected object boxes from inference
        track_boxes = trackResults[0].boxes.cpu().numpy()

        #Get the detected object boxes ids from tracked inference
        trackIds = []
        if trackResults[0].boxes.id is None:
            pass
        else:
            trackIds = trackResults[0].boxes.id.cpu().numpy().astype(int)

        #Retrieve properties about inferred result, check if it detects a face (face class stored in cls variable as 0)
        for box, track_id in zip(track_boxes, trackIds):
            cls = int(box.cls[0])
            #If a person's face is detected in the detection box,
            if cls == 0:
                #Then iterate for each existing instances of boxes / people:
                for trackId in trackIds:
                    #Save the annotated frame as an image file in a local directory.
                    current_time = time.strftime("%H-%M", time.localtime())
                    filename = "person_" + str(trackId) + "_" + current_time + ".jpg"
                    absolute_path = os.path.join(os.getcwd(), 'results', filename)
                    print(absolute_path)
                    cv2.imwrite(absolute_path, annotated_frame)

        #Show the video frame in screen for demonstration purposes
        cv2.imshow("Real Time Tracking, Press Q to Quit", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print("Cam stream ended/not found")
        break

cam.release()
cv2.destroyAllWindows()

