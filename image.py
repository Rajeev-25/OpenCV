import cv2
import dlib
import numpy as np

# Initialize the face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('D:\shape_predictor_68_face_landmarks.dat')

# Open the video capture device
cap = cv2.VideoCapture(0)

# Create a canvas to draw on
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Define the landmarks corresponding to the mouth
mouth_landmarks = [48, 54]

# Define a reference distance for comparison
ref_distance = 3.0  # in cm

while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_detector(gray)

    for face in faces:
        # Predict the facial landmarks for the detected face
        shape = shape_predictor(gray, face)

        # Extract the mouth landmarks from the facial landmarks
        mouth_pts = [(shape.part(idx).x, shape.part(idx).y) for idx in mouth_landmarks]

        # Calculate the distance between the two mouth landmarks
        distance = np.sqrt((mouth_pts[0][0] - mouth_pts[1][0]) ** 2 +
                           (mouth_pts[0][1] - mouth_pts[1][1]) ** 2)

        # If the distance is greater than the reference distance, apply Gaussian blur to the frame
        if distance > ref_distance:
            frame = cv2.GaussianBlur(frame, (25, 25), 0)

        # Draw circles on the mouth landmarks
        for pt in mouth_pts:
            cv2.circle(frame, pt, 2, (0, 0, 255), -1)

    # Display the frame
    cv2.imshow('Mouth Recognition', frame)

    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and destroy any remaining windows
cap.release()
cv2.destroyAllWindows()
