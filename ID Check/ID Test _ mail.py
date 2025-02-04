import cv2
import tkinter as tk
from tkinter import Label, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

# Load YOLOv8 model for ID tag detection
model = YOLO("runs/detect/train14/weights/best.pt")

# Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize Tkinter window
root = tk.Tk()
root.title("ID Detection & Face Recognition")

# Label to display the video feed
panel = Label(root)
panel.pack()

# Counter to track missing ID instances
missing_id_count = 0
face_detection_active = False  # Flag to activate face detection

# Open webcam
cap = cv2.VideoCapture(0)

def process_frame():
    global missing_id_count, face_detection_active

    ret, frame = cap.read()
    if not ret:
        root.after(10, process_frame)
        return

    # Run YOLO on the frame (for ID tag detection)
    results = model.predict(source=frame, conf=0.3)
    found_id = False  # Flag to check if ID is detected

    for box in results[0].boxes:
        cls = int(box.cls[0])  # Get class ID
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates

        if cls == 0:  # Assuming class 0 is ID tag
            found_id = True
            # Draw bounding box around ID tag
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "ID", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # If ID is not found, increase the missing count
    if not found_id:
        missing_id_count += 1
        print(f"ID tag missing! Count: {missing_id_count}")
    else:
        missing_id_count = 0  # Reset counter if ID is found

    # If ID is missing for 3 instances, activate face detection
    if missing_id_count >= 3:
        face_detection_active = True

    if face_detection_active:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Draw bounding box around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Sahana", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if len(faces) > 0:
            messagebox.showinfo("Alert", "Mail Sent to Prof. Maareeswari")
            missing_id_count = 0  # Reset counter after face detection
            face_detection_active = False  # Stop face detection

    # Convert frame to RGB format for Tkinter display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img = img.resize((600, 450))  # Resize for better visibility

    img_tk = ImageTk.PhotoImage(img)
    panel.config(image=img_tk)
    panel.image = img_tk  # Prevent garbage collection

    root.after(10, process_frame)  # Schedule next frame processing

# Start button
btn_start = tk.Button(root, text="Start Camera", command=process_frame, font=("Arial", 14))
btn_start.pack(pady=10)

root.mainloop()

cap.release()
cv2.destroyAllWindows()
