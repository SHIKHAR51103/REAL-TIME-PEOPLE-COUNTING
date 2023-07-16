import cv2

# Load the pre-trained Haar cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

# Initialize the person count
person_count = 0

while True:
    # Read each frame from the video capture
    ret, frame = video_capture.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Detect upper bodies in the frame
    upper_bodies = upper_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Combine the detected faces and upper bodies
    detections = list(faces) + list(upper_bodies)
    
    # Draw rectangles around the detected regions
    for (x, y, w, h) in detections:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Update the person count
    person_count = len(detections)
    
    # Display the person count on the frame
    cv2.putText(frame, f'Person Count: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow('Real-time People Counting', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
