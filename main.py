from deepface import DeepFace
import cv2
from matplotlib import pyplot as plt

def process_frame(frame):
    # Use DeepFace for facial emotion recognition
    try:
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Extract emotion information
        emotion = results[0]['dominant_emotion']  # Access the first result in the list

        # Display the predicted emotion
        print(f'Emotion: {emotion}')

    except ValueError as e:
        # Handle the case when no face is detected
        print("Error:", e)

    # Display the resulting frame
    cv2.imshow('Facial Emotion Detection', frame)

# User choice: '1' for live webcam, '2' for image file
user_choice = input("Choose an option:\n1. Live Webcam\n2. Image from Device\nEnter choice (1/2): ")

if user_choice == '1':
    # Live webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # Process the frame
        process_frame(frame)

        # Break the loop if 'q' key is pressed
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # 'q' key or ESC key
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

elif user_choice == '2':
    # Image from device
    image_path = input("Enter the path of the image file: ")
    
    # Read the image
    image = cv2.imread(image_path)

    # Process the image
    process_frame(image)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Invalid choice. Please choose either '1' or '2'.")