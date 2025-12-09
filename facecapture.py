import cv2
import os
import time

# Load the pre-trained face classifier from OpenCV
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize the webcam video capture
video_capture = cv2.VideoCapture(0)

# Create a folder to save captured faces
output_folder = "captured_faces"
os.makedirs(output_folder, exist_ok=True)

def detect_and_save_faces(vid, frame_count):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        # Extract the face from the frame
        face = vid[y:y+h, x:x+w]
        # Save the face image to the output folder
        face_filename = os.path.join(output_folder, f"face_{frame_count}.jpg")
        cv2.imwrite(face_filename, face)
    return faces

# Get the start time
start_time = time.time()

frame_count = 0
while True:
    # Capture frame-by-frame from the webcam
    result, video_frame = video_capture.read()
    if not result:
        break
    
    # Detect faces and draw bounding boxes, also save faces
    faces = detect_and_save_faces(video_frame, frame_count)
    frame_count += 1
    
    # Display the resulting frame
    cv2.imshow("My Face Detection Project", video_frame)
    
    # Break the loop if 'q' key is pressed or if 2 seconds have passed
    if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time) > 2:
        break

# Release the video capture object and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
