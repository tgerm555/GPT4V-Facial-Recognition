from functions import GPT4V
import face_recognition
import cv2
import numpy as np
import os

video_capture = cv2.VideoCapture(1)
training_images_dir = "training_images"
known_face_encodings = []
known_face_names = []

for filename in os.listdir(training_images_dir):
    if filename.endswith(".jpeg") or filename.endswith(".jpg"):

        image_path = os.path.join(training_images_dir, filename)
        image = face_recognition.load_image_file(image_path)

        face_encoding = face_recognition.face_encodings(image)[0]

        known_face_encodings.append(face_encoding)
        known_face_names.append("Tyler Germain")

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:

    ret, frame = video_capture.read()

    if process_this_frame:

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        screenshot_filename = "screenshot.jpg"
        cv2.imwrite(screenshot_filename, frame)
        print("Screenshot saved as", screenshot_filename)
        people = ",".join(str(x) for x in face_names)
        response = GPT4V(people, screenshot_filename)
        print(response)