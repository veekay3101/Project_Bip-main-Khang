import cv2
import cvzone
import math
import threading
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
import time
import pygame
import torch

# Initialize pygame to play the alert sound
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.wav")

# Load YOLO model (use yolov8n.pt for better real-time performance)
model = YOLO('yolov8n.pt')
if torch.cuda.is_available():
    model.to('cuda')
classnames = model.names

# GUI setup
root = tk.Tk()
root.title("Real-Time Fall Detection")
root.geometry("800x600")

label_video = tk.Label(root)
label_video.pack()

label_warning = tk.Label(root, text="", font=("Arial", 24), fg="red")
label_warning.pack(pady=10)

status_bar = tk.Label(root, text="Monitoring...", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# Use webcam (0 = default camera)
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

prev_time = 0
fall_timer = 0
frame_skip = 1
frame_count = 0

def process_frame():
    global prev_time, fall_timer, frame_count

    ret, frame = cap.read()
    if not ret:
        status_bar.config(text="Camera not detected or disconnected.")
        return

    frame_count += 1
    if frame_count % frame_skip != 0:
        root.after(1, process_frame)
        return

    frame = cv2.resize(frame, (640, 480))
    results = model(frame, verbose=False)

    fall_detected = False

    for info in results:
        boxes = info.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            class_name = classnames.get(cls_id, f"Unknown({cls_id})")
            confidence = math.ceil(conf * 100)

            if class_name != 'person' or confidence < 80:
                continue

            width = x2 - x1
            height = y2 - y1
            is_fall = width > height

            color = (0, 0, 255) if is_fall else (0, 255, 0)
            cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6, colorC=color)
            cvzone.putTextRect(frame, f'{class_name} {confidence}%', [x1 + 8, y1 - 12], thickness=2, scale=1)

            if is_fall:
                fall_detected = True
                cvzone.putTextRect(frame, ' FALL DETECTED ', [x1, y2 + 20], scale=2, thickness=2, colorR=(0, 0, 255))

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if fall_detected:
        label_warning.config(text="FALL DETECTED")
        status_bar.config(text="Warning: Fall detected!")
        if time.time() - fall_timer > 5:
            pygame.mixer.Sound.play(alert_sound)
            fall_timer = time.time()
    else:
        label_warning.config(text="")
        status_bar.config(text="Monitoring...")

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)

    label_video.imgtk = img_tk
    label_video.configure(image=img_tk)

    root.after(10, process_frame)

# Start real-time processing
threading.Thread(target=process_frame).start()

root.mainloop()
cap.release()
cv2.destroyAllWindows()
