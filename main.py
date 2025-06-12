import cv2
import numpy as np
import os
import time

# Constants
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Paths to the models and Haar Cascade file
MODEL_DIR = os.path.join(os.getcwd(), 'data')
GENDER_MODEL_PATH = os.path.join(MODEL_DIR, 'deploy_gender.prototxt')
GENDER_WEIGHTS_PATH = os.path.join(MODEL_DIR, 'gender_net.caffemodel')
HAAR_CASCADE_PATH = os.path.join(MODEL_DIR, 'haarcascade_frontalface_default.xml')

# Function to load models
def load_models():
    try:
        # Check if required files exist
        if not os.path.exists(GENDER_MODEL_PATH) or not os.path.exists(GENDER_WEIGHTS_PATH):
            raise FileNotFoundError(f"Gender model files not found in {MODEL_DIR}")
        if not os.path.exists(HAAR_CASCADE_PATH):
            raise FileNotFoundError(f"Haar Cascade file not found at {HAAR_CASCADE_PATH}")
        
        # Load the models
        gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL_PATH, GENDER_WEIGHTS_PATH)
        face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        return gender_net, face_cascade
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

# Function to predict gender
def predict_gender(face_img, gender_net):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    gender_net.setInput(blob)
    predictions = gender_net.forward()
    return GENDER_LIST[np.argmax(predictions[0])]

# Real-time detection function
def real_time_detection():
    gender_net, face_cascade = load_models()
    if gender_net is None or face_cascade is None:
        print("Failed to load models. Exiting...")
        return

    # Start video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("Starting Real-Time Gender Detection...")
    while True:
        start_time = time.time()

        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            if face.size == 0:
                continue

            # Predict gender
            try:
                gender = predict_gender(face, gender_net)
            except Exception as e:
                gender = "Unknown"
                print(f"Prediction error: {e}")

            # Draw rectangle and overlay text
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 100), 2)
            cv2.putText(frame, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 144, 0), 2)

        # Calculate latency
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        cv2.putText(frame, f"Latency: {latency:.2f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display frame
        cv2.imshow('Real-Time Gender Detection', frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_detection()
