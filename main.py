import cv2
import torch
import numpy as np
import pygame
import face_recognition
import firebase_admin
from firebase_admin import credentials, db
import ultralytics

# Path to the alarm sound
path_alarm = "Alarm/alarm.wav"

# Initialize pygame
pygame.init()
pygame.mixer.music.load(path_alarm)

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-recognition-sss-default-rtdb.firebaseio.com/",
    'storageBucket': "gs://face-recognition-sss.appspot.com"
})

# Reference to the database
ref = db.reference('known_faces')

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load video
cap = cv2.VideoCapture(0)  # Using webcam, change to video file if needed
target_classes = ['car', 'bus', 'truck', 'person']

# Polygon points
pts = []

# Function to draw polygon (ROI)
def draw_polygon(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        pts.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        pts = []

# Function to check if a point is inside a polygon
def inside_polygon(point, polygon):
    if len(polygon) < 3:
        return False  # Not a valid polygon
    polygon = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(polygon, (point[0], point[1]), False)
    return result >= 0

cv2.namedWindow('Video')
cv2.setMouseCallback('Video', draw_polygon)

def preprocess(img):
    height, width = img.shape[:2]
    ratio = height / width
    img = cv2.resize(img, (640, int(640 * ratio)))
    return img

def download_known_faces():
    known_face_encodings = []
    known_face_names = []
    faces = ref.get()
    for key, face_data in faces.items():
        known_face_encodings.append(np.array(face_data['encoding']))
        known_face_names.append(face_data['name'])
    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = download_known_faces()

alarm_playing = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_detected = frame.copy()
    frame = preprocess(frame)

    results = model(frame)

    person_detected = False
    face_recognized = False

    for index, row in results.pandas().xyxy[0].iterrows():
        center_x = None
        center_y = None

        if row['name'] in target_classes:
            name = str(row['name'])
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
            # Write name
            cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            # Draw center
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            if name == 'person':
                person_detected = True
                face_location = [(y1, x2, y2, x1)]
                face_encodings = face_recognition.face_encodings(frame_detected, face_location)

                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        recognized_name = known_face_names[best_match_index]
                        face_recognized = True
                        cv2.putText(frame, f"Recognized: {recognized_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        break

    if len(pts) >= 4:
        frame_copy = frame.copy()
        cv2.fillPoly(frame_copy, [np.array(pts, dtype=np.int32)], (0, 255, 0))
        frame = cv2.addWeighted(frame_copy, 0.1, frame, 0.9, 0)

    if person_detected and not face_recognized:
        for index, row in results.pandas().xyxy[0].iterrows():
            if row['name'] == 'person':
                center_x = int((row['xmin'] + row['xmax']) / 2)
                center_y = int((row['ymin'] + row['ymax']) / 2)
                if inside_polygon((center_x, center_y), pts):
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play()
                        alarm_playing = True
                    cv2.putText(frame, "Target", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, "Person Detected", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.rectangle(frame, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), (0, 0, 255), 3)
                    break
    else:
        if alarm_playing:
            pygame.mixer.music.stop()
            alarm_playing = False

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
