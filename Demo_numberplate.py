import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk
import easyocr
import numpy as np

# Load the trained YOLOv8 model
model = YOLO("best.pt")

# Initialize the EasyOCR reader
ocr_reader = easyocr.Reader(['en'])

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
    
    # Display the annotated image
    show_image(annotated_image_rgb)
    
    # Extract the number plate section and apply OCR
    extract_number_plate(results[0].boxes, results[0].names, annotated_image_rgb)  # Pass bounding boxes, names, and annotated image

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

# Function to extract and OCR the number plate using edge detection, sharpening, thresholding, and contrast enhancement
# Function to extract and OCR the number plate using edge detection, sharpening, thresholding, contrast enhancement, and shadow removal
def extract_number_plate(boxes, names, annotated_image):
    # Iterate over each detected bounding box
    for i, box in enumerate(boxes):
        label = names[int(box.cls)]  # Get the label of the current detection
        print(f"Detected label: {label}")  # Debugging: Print detected label
        if label == "NumberPlate":  # Check if the label is "NumberPlate"
            # Get the coordinates of the bounding box (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Debugging: Print the bounding box coordinates
            print(f"Bounding box coordinates (x1, y1, x2, y2): ({x1}, {y1}, {x2}, {y2})")

            # Crop the image to the bounding box (number plate section)
            cropped_image = annotated_image[y1:y2, x1:x2]

            # Ensure the cropped region is not empty
            if cropped_image.size == 0:
                print("Error: Cropped image is empty.")
                continue

            # Show the cropped image of the number plate (ROI)
            show_cropped_image(cropped_image)

            # 1. Enlarge the cropped image
            enlarged_image = cv2.resize(cropped_image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)  # Enlarging the image by a factor of 2

            # Show the enlarged image (optional for debugging)
            #cv2.imshow("Enlarged Image", enlarged_image)

            # 2. Convert the enlarged image to grayscale
            gray = cv2.cvtColor(enlarged_image, cv2.COLOR_BGR2GRAY)

            # 3. Remove shadows using morphological transformations
            # Create a kernel for morphological operation
            kernel = np.ones((5, 5), np.uint8)
            shadow_removed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)  # Closing operation to remove small shadows

            # Show the shadow removed image (optional for debugging)
            #cv2.imshow("Shadow Removed Image", shadow_removed)

            # 4. Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Parameters can be adjusted for better results
            contrast_enhanced_image = clahe.apply(shadow_removed)

            # Show the contrast-enhanced image (optional for debugging)
            #cv2.imshow("Contrast Enhanced Image", contrast_enhanced_image)

            # 5. Sharpen the contrast-enhanced grayscale image
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Simple sharpening kernel
            sharpened_image = cv2.filter2D(contrast_enhanced_image, -1, kernel)  # Apply the sharpening filter

            # Show the sharpened image (optional for debugging)
            #cv2.imshow("Sharpened Image", sharpened_image)

            # 6. Apply thresholding (Binary thresholding)
            _, thresholded_image = cv2.threshold(sharpened_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Show the thresholded image (optional for debugging)
            #cv2.imshow("Thresholded Image", thresholded_image)

            # 7. Apply edge detection (Canny edge detection) after thresholding
            edges = cv2.Canny(thresholded_image, 100, 200)  # Adjust thresholds (100, 200) as needed

            # Show the edges image (optional for debugging)
            #cv2.imshow("Edge Detected Image", edges)

            # 8. Resize the edge-detected image for better OCR accuracy (optional)
            resized_edges = cv2.resize(edges, (600, 200))  # Resizing to a more OCR-friendly size

            # Use EasyOCR to extract text from the thresholded image
            ocr_result = ocr_reader.readtext(resized_edges)

            # If OCR result is found, display the detected text
            if ocr_result:
                for detection in ocr_result:
                    text = detection[1]
                    print(f"Detected number plate: {text}")
                    # Display the OCR result on the image (optional)
                    cv2.putText(sharpened_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Optionally show the cropped image with OCR text (for debugging)
            cv2.imshow("Cropped Image with OCR", sharpened_image)
            cv2.waitKey(0)



# Function to show the cropped number plate image (ROI)
def show_cropped_image(cropped_image):
    try:
        # Convert the cropped image to RGB for display
        cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

        # Convert the numpy image array (RGB) to a PIL image
        img = Image.fromarray(cropped_image_rgb)
        img = img.resize((400, 200))  # Resize for display

        # Convert the PIL image to ImageTk format for Tkinter
        img_tk = ImageTk.PhotoImage(img)
        
        # Display the cropped image in the Tkinter window
        cropped_panel.config(image=img_tk)
        cropped_panel.image = img_tk  # Prevent garbage collection
    except Exception as e:
        print(f"Error displaying cropped image: {e}")

# Create GUI window
root = tk.Tk()
root.title("YOLOv8 Number Plate Detection with Edge Detection and OCR")

# String variable to hold the selected image file path
image_path = tk.StringVar()

# Button to select an image file
btn_select = tk.Button(root, text="Select Image", command=select_file, font=("Arial", 14))
btn_select.pack(pady=10)

# Label to display the image
panel = tk.Label(root)
panel.pack(pady=10)

# Label to display the cropped number plate
cropped_panel = tk.Label(root)
cropped_panel.pack(pady=10)

# Bind the Enter key to trigger YOLO inference
root.bind('<Return>', run_yolo)

# Run Tkinter event loop
root.mainloop()
