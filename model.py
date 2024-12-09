import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Define image size and path to your dataset
img_size = 300
dataset_path = "Data"  # Path to your dataset directory

# Initialize the ImageDataGenerator for loading images
datagen = ImageDataGenerator(rescale=1./255)  # Normalize images to [0, 1]

# Load images from the dataset folder
train_data = datagen.flow_from_directory(
    dataset_path,  # Main directory where 'hello', 'thankyou' folders are located
    target_size=(img_size, img_size),  # Resize images to a consistent size
    batch_size=32,
    class_mode='sparse'  # Use 'sparse' for integer labels
)

# Print the class labels (folder names)
print("Class indices:", train_data.class_indices)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_data.class_indices), activation='softmax')  # Output layer for classification
])

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary to check the architecture
model.summary()

# Callback for saving the best model during training
checkpoint = ModelCheckpoint('asl_model_best1.keras', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

# Train the model using the data from the generator
model.fit(train_data, epochs=20, callbacks=[checkpoint])

# Save the final trained model
model.save("asl_model_final1.h5")
print("Model training complete and saved!")
