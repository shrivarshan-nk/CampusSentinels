import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk
import os

# Load the trained YOLOv8 model
model = YOLO("runs/detect/train3/weights/best.pt")

# Function to select a file (single image)
def select_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    if file_path:
        image_path.set(file_path)

# Function to run YOLO on the selected image and display result
def run_yolo(event=None):
    image_path_value = image_path.get()
    if not image_path_value:
        print("No image selected")
        return
    
    print(f"Running inference on {image_path_value}...")
    
    # Run YOLO model prediction on the selected image
    results = model.predict(source=image_path_value, conf=0.5)
    
    # Extract the path of the processed image (single image prediction)
    output_image_path = results[0].path  # Use the 'path' attribute to get the image path
    
    print(f"Processed image: {output_image_path}")
    
    # Annotate the image with bounding boxes and labels
    annotated_image = results[0].plot()  # Plot the detection results on the image
    
    # Convert the annotated image from BGR (OpenCV) to RGB (PIL)
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    # Convert the annotated image (now in RGB) to PIL format
    show_image(annotated_image_rgb)

# Function to display an image in the Tkinter window
def show_image(image_array):
    try:
        # Convert the numpy image array (RGB) to a PIL image
        img = Image.fromarray(image_array)
        img = img.resize((500, 400))  # Resize for display

        # Convert the PIL image to ImageTk format for Tkinter
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk  # Prevent garbage collection
    except Exception as e:
        print(f"Error displaying image: {e}")

# Create GUI window
root = tk.Tk()
root.title("YOLOv8 Image Testing")

# String variable to hold the selected image file path
image_path = tk.StringVar()

# Button to select an image file
btn_select = tk.Button(root, text="Select Image", command=select_file, font=("Arial", 14))
btn_select.pack(pady=10)

# Label to display the image
panel = tk.Label(root)
panel.pack()

# Bind the Enter key to trigger YOLO inference
root.bind('<Return>', run_yolo)

# Run Tkinter event loop
root.mainloop()
