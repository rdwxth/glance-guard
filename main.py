import cv2
import numpy as np
from screen_brightness_control import set_brightness

def get_face_detector():
    # Load pre-trained Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

def detect_face(img, face_cascade):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    return faces

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Initialize face detector
    face_cascade = get_face_detector()


    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Detect faces
        faces = detect_face(frame, face_cascade)

        if len(faces) > 0:
            # If faces are detected, set brightness to maximum
            set_brightness(100)
        else:
            # If no faces are detected, dim the monitor
            set_brightness(50)

        # Display the resulting frame
        cv2.imshow('Webcam Feed', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
