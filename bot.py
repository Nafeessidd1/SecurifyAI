import ssl
import cv2
import torch
import numpy as np
import pygame
from PIL import Image
import telegram

ssl._create_default_https_context = ssl._create_unverified_context

# Path to the alarm sound
path_alarm = "/Users/nafeessiddiqui/Desktop/Projects/Smart Surveillance System/Alarm/alarm.wav"

# Initialize pygame for alarm sound
pygame.init()
pygame.mixer.music.load(path_alarm)

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize video capture
cap = cv2.VideoCapture("Test Videos/thief_video2.mp4")
target_classes = ['car', 'bus', 'truck', 'person']

# Initialize counter and number of photos
count = 0
number_of_photos = 3

# Polygon points
pts = []

# Initialize Telegram bot
bot_token = '6715814880:AAHUJFeLaT3BFU8arCxoBPRZpAezAP6PLBE'
chat_id = 'SmartcctvYolo_bot'
bot = telegram.Bot(token=bot_token)

# Function to send Telegram notification
def send_telegram_notification(message):
    bot.send_message(chat_id=chat_id, text=message)

# Function to preprocess image
def preprocess(img):
    height, width = img.shape[:2]
    ratio = height / width
    img = cv2.resize(img, (640, int(640 * ratio)))
    return img

# Main loop for processing video frames
while True:
    ret, frame = cap.read()
    if ret:
        frame_detected = frame.copy()
        frame = preprocess(frame)
        results = model(frame)

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

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
                cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            if len(pts) >= 4:
                frame_copy = frame.copy()
                cv2.fillPoly(frame_copy, np.array([pts]), (0, 255, 0))
                frame = cv2.addWeighted(frame_copy, 0.1, frame, 0.9, 0)
                if center_x is not None and center_y is not None:
                    if inside_polygon((center_x, center_y), np.array([pts])) and name == 'person':
                        mask = np.zeros_like(frame_detected)
                        points = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])
                        points = points.reshape((-1, 1, 2))
                        mask = cv2.fillPoly(mask, [points], (255, 255, 255))
                        frame_detected = cv2.bitwise_and(frame_detected, mask)
                        if count < number_of_photos:
                            cv2.imwrite("Detected Photos/detected" + str(count) + ".jpg", frame_detected)
                        if not pygame.mixer.music.get_busy():
                            pygame.mixer.music.play()
                        # Send Telegram notification
                        send_telegram_notification("Person detected!")
                        cv2.putText(frame, "Target", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(frame, "Person Detected", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        count += 1
        cv2.imshow("Video", frame)
    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
