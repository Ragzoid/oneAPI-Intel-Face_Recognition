#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 21:03:04 2023

@author: raghul
"""
import onednn as dnn
import numpy as np
from sklearn.model_selection import train_test_split
import csv
from datetime import datetime

# Load and preprocess your dataset
X = np.load("path/to/dataset.npy")
y = np.load("path/to/labels.npy")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model architecture
model = dnn.Model()
model.add(dnn.Convolution2D(in_channels=1, out_channels=32, kernel_size=(3, 3), activation="relu"))
model.add(dnn.MaxPooling2D(pool_size=(2, 2)))
model.add(dnn.Convolution2D(in_channels=32, out_channels=64, kernel_size=(3, 3), activation="relu"))
model.add(dnn.MaxPooling2D(pool_size=(2, 2)))
model.add(dnn.Flatten())
model.add(dnn.Dense(units=128, activation="relu"))
model.add(dnn.Dense(units=num_classes, activation="softmax"))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Convert labels to one-hot encoding
num_classes = len(np.unique(y_train))
y_train_encoded = dnn.utils.to_categorical(y_train, num_classes)
y_test_encoded = dnn.utils.to_categorical(y_test, num_classes)

# Train the model
model.fit(X_train, y_train_encoded, batch_size=32, epochs=10, validation_data=(X_test, y_test_encoded))

# Save the trained model
model.save("path/to/trained_model.xml", "path/to/trained_model.bin")

# Perform inference and update attendance
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Preprocess the frame
    preprocessed_frame = preprocess_image(frame)

    # Perform inference
    output = model.run([preprocessed_frame])
    predicted_class = np.argmax(output)

    # Get the predicted class label and update attendance
    predicted_label = class_labels[predicted_class]
    update_attendance(predicted_label)

    # Display the result on the frame
    cv2.putText(frame, "Attendance: " + predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Attendance Management System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()

# Function to preprocess frame
def preprocess_image(frame):
    # Perform necessary preprocessing steps
    preprocessed_frame = ...

    return preprocessed_frame

# Function to update attendance record
def update_attendance(label):
    # Update attendance record
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d")
    time_string = now.strftime("%H:%M:%S")

    with open("path/to/attendance.csv", "a") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([label, date_string, time_string])
