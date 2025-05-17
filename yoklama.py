import cv2
from simple_facerec import SimpleFacerec
from datetime import datetime
from datetime import date

# Get today's date
today = date.today()
# Convert the date to a specific string format, e.g., "Apr-22-2025"
day = today.strftime("%b-%d-%Y")
# Create the attendance file name
day_str = "yoklama" + day + ".csv"
print(day_str)

# Open the attendance file and write the header row
dosya = open(day_str, "a")
dosya.write("Name,Time")
dosya.close()

# Function to write student name and time into the attendance file
def yoklamayaYaz(name):
    with open(day_str, 'r+') as f:
        # Read existing data
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')  # Split the line by comma
            nameList.append(entry)   # Add to list

        # If the person is not already recorded, add them (to avoid duplicates)
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')  # Get the current time
            f.writelines(f'\n{name},{dtString}')  # Write to file

# Initialize the SimpleFacerec class
sfr = SimpleFacerec()
# Load previously saved face encodings (from the 'images' folder)
sfr.load_encoding_images("images/")

# Start the webcam (0: default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
   
    # Detect faces and get their names in the frame
    face_locations, face_names = sfr.detect_known_faces(frame)

    # For each detected face:
    for face_loc, name in zip(face_locations, face_names):
        # Get the coordinates of the face
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        # Display the name above the face
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
         
        # Save the person to the attendance file
        yoklamayaYaz(name)

    # Show the frame on the screen
    cv2.imshow("frame", frame)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()
# Close all OpenCV windows
cv2.destroyAllWindows()
