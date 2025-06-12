import cv2  # OpenCV library for computer vision tasks
import numpy as np  # Library for numerical operations
import os  # Library to interact with the operating system
import time  # Library for time-related operations
import pyttsx3  # Library for text-to-speech conversion
import threading  # Library for concurrent execution

# Constants
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)  # Mean values for pre-processing images for gender detection
GENDER_LIST = ['Male', 'Female'] # List of gender labels
FRAME_WIDTH = 640  # Width of the video frame
FRAME_HEIGHT = 480  # Height of the video frame
PROCESS_FRAME_RATE = 5  # Only process every 5th frame for efficiency

# Paths to the models and Haar Cascade file
MODEL_DIR = os.path.join(os.getcwd(), 'data')  # Directory containing model files
GENDER_MODEL_PATH = os.path.join(MODEL_DIR, 'deploy_gender.prototxt')  # Path to gender model configuration
GENDER_WEIGHTS_PATH = os.path.join(MODEL_DIR, 'gender_net.caffemodel')  # Path to gender model weights
HAAR_CASCADE_PATH = os.path.join(MODEL_DIR, 'haarcascade_frontalface_default.xml')  # Path to Haar Cascade for face detection

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()  # Initialize the TTS engine
tts_engine.setProperty('rate', 150)  # Set speaking rate
tts_engine.setProperty('volume', 0.9)  # Set volume level

def speak(text):
    """Speak the given text using the TTS engine."""
    tts_engine.say(text)  # Add text to the speaking queue
    tts_engine.runAndWait()  # Process the speaking queue

def load_models():
    """Load the gender detection model and Haar Cascade for face detection."""
    try:
        # Check if gender model files exist
        if not os.path.exists(GENDER_MODEL_PATH) or not os.path.exists(GENDER_WEIGHTS_PATH):
            raise FileNotFoundError(f"Gender model files not found in {MODEL_DIR}")
        # Check if Haar Cascade file exists
        if not os.path.exists(HAAR_CASCADE_PATH):
            raise FileNotFoundError(f"Haar Cascade file not found at {HAAR_CASCADE_PATH}")
        
        # Load gender detection model
        gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL_PATH, GENDER_WEIGHTS_PATH)
        # Load Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        return gender_net, face_cascade
    except Exception as e:
        print(f"Error loading models: {e}")  # Log any errors during model loading
        return None, None

def predict_gender(face_img, gender_net):
    """
    Predict the gender of a face using the gender detection model.
    Args:
        face_img: Cropped image of the detected face.
        gender_net: Pre-trained gender detection model.
    Returns:
        Predicted gender as 'Male' or 'Female'.
    """
    # Pre-process the face image
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    # Set the input to the network
    gender_net.setInput(blob)
    # Get predictions from the network
    predictions = gender_net.forward()
    # Return the gender with the highest probability
    return GENDER_LIST[np.argmax(predictions[0])]

def real_time_detection():
    """Perform real-time gender detection using webcam input."""
    # Load models for gender detection and face detection
    gender_net, face_cascade = load_models()
    if gender_net is None or face_cascade is None:
        print("Failed to load models. Exiting...")
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0)  # Capture video from the default webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)  # Set video frame width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)  # Set video frame height

    last_gender = None  # Store the last detected gender
    frame_count = 0  # Counter for frames processed

    print("Starting Real-Time Gender Detection...")
    speak("Starting real-time gender detection.")  # Notify user via TTS

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_count += 1

        if frame_count % PROCESS_FRAME_RATE == 0:  # Process every 5th frame
            # Resize the frame to half size for faster face detection
            small_frame = cv2.resize(gray_frame, (0, 0), fx=0.5, fy=0.5)
            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(small_frame, scaleFactor=1.1, minNeighbors=5)
            # Adjust coordinates to match the original frame size
            faces = [(int(x*2), int(y*2), int(w*2), int(h*2)) for (x, y, w, h) in faces]

            for (x, y, w, h) in faces:
                # Extract the face region
                face = frame[y:y + h, x:x + w]
                if face.size == 0:  # Skip empty faces
                    continue

                # Predict gender and speak if it changes
                try:
                    gender = predict_gender(face, gender_net)
                    if gender != last_gender:  # Speak only if gender changes
                        speak(f"Detected {gender}")
                        last_gender = gender
                except Exception as e:
                    print(f"Prediction error: {e}")
                    gender = "Unknown"

                # Draw a rectangle around the face and display the gender
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 100), 2)
                cv2.putText(frame, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 144, 0), 2)

        # Add timestamp overlay to the frame
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        cv2.putText(frame, f"Timestamp: {timestamp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the video frame with overlays
        cv2.imshow('Real-Time Gender Detection', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            speak("Exiting real-time gender detection. Goodbye!")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Start the real-time gender detection process
    real_time_detection()
