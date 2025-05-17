import glob  # Used to find file paths (e.g., to get all images in a folder)
import os  # Used for file and folder operations
import cv2  # OpenCV library, used for image processing
import face_recognition  # Powerful library used for face recognition
import numpy as np  # Used for numerical operations and arrays

class SimpleFacerec:  # Class definition to handle face recognition operations
    def __init__(self):  # Constructor method that runs when the class is initialized
        self.known_face_encodings = []  # Stores the vector representations of known faces
        self.known_face_names = []  # Stores the names corresponding to the faces
        self.frame_resizing = 0.25  # Scaling factor to resize frame (to increase processing speed)

    def load_encoding_images(self, images_path):  # Loads image files from the given folder and extracts face encodings
        images_path = glob.glob(os.path.join(images_path, "*.*"))  # Gets all files in the given folder
        print("{} encoding images found.".format(len(images_path)))  # Prints the number of images found

        for img_path in images_path:  # Processes each image
            img = cv2.imread(img_path)  # Reads the image file
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converts BGR (default OpenCV format) to RGB

            basename = os.path.basename(img_path)  # Gets the file name (excluding the path)
            filename, ext = os.path.splitext(basename)  # Splits the file name and its extension

            img_encoding = face_recognition.face_encodings(rgb_img)  # Extracts face encodings from the image
            if img_encoding:  # If a face is detected (not an empty list)
                self.known_face_encodings.append(img_encoding[0])  # Appends the first detected face encoding to the list
                self.known_face_names.append(filename)  # Adds the file name (used as the person's name) to the list

        print("Encoding images loaded")  # Message indicating that face encodings are loaded

    def detect_known_faces(self, frame):  # Detects known faces in a real-time video frame
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)  # Resizes the frame (for speed)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # Converts the small frame to RGB

        face_locations = face_recognition.face_locations(rgb_small_frame)  # Finds face locations in the image
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)  # Gets encodings for detected faces

        face_names = []  # List to store names of recognized faces
        for face_encoding in face_encodings:  # For each face encoding
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)  # Compares with known faces
            name = "Unknown"  # Default name if face is not recognized

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)  # Calculates distance (similarity)
            if len(face_distances) > 0:  # If there are known faces to compare
                best_match_index = np.argmin(face_distances)  # Gets the index of the closest match
                if matches[best_match_index]:  # If the best match is valid
                    name = self.known_face_names[best_match_index]  # Gets the corresponding name
            face_names.append(name)  # Adds the name to the result list

        face_locations = np.array(face_locations)  # Converts face locations to a numpy array
        face_locations = (face_locations / self.frame_resizing).astype(int)  # Rescales coordinates to original frame size

        return face_locations, face_names  # Returns the face locations and recognized names













