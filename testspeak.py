import cv2
from cvzone.HandTrackingModule import HandDetector
from keras.models import load_model
import numpy as np
import math
import pyttsx3  # For text-to-speech
import time  # To add delay

# Load the pre-trained model
model = load_model(r"C:\Users\dasar\OneDrive\Desktop\asl\asl_model_final2.h5")

# Load labels
labels = ["Hello", "Thank_you", "ya"]

# Initialize HandDetector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Parameters for cropping and resizing the image
offset = 20
imgSize = 300
stable_label = ""  # To track the stable gesture
start_time = 0  # Timer to track stability

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white canvas to resize and fit the image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand from the image
        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        # Prepare image for prediction (normalize and expand dimensions)
        imgArray = np.array(imgWhite) / 255.0  # Normalize pixel values to [0, 1]
        imgArray = np.expand_dims(imgArray, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(imgArray)
        
        # Get the class with the highest probability
        index = np.argmax(predictions[0])
        label = labels[index]

        # Check for gesture stability over time (5 ms threshold)
        if stable_label == label:
            if time.time() - start_time >= 0.005:  # 5 ms threshold
                engine.say(label)
                engine.runAndWait()
                start_time = time.time()  # Reset timer after speaking
        else:
            stable_label = label
            start_time = time.time()  # Reset timer for a new gesture

        # Display predicted label on the image
        cv2.rectangle(imgOutput, (x - offset, y - offset - 70), 
                      (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, label, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), 
                      (x + w + offset, y + h + offset), (0, 255, 0), 4)

        # Show cropped hand and resized white image
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    # Show the main output image with predictions
    cv2.imshow('Image', imgOutput)
    
    # Wait for the 'q' key to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
