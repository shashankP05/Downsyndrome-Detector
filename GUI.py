import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
from mtcnn import MTCNN

# Load model
model = tf.keras.models.load_model("downsyndrome_classifier.keras")
CLASS_NAMES = ['Down Syndrome', 'Healthy']

# Initialize MTCNN detector
detector = MTCNN()

# GUI setup
top = tk.Tk()
top.geometry('800x600')
top.title('Down Syndrome Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

def Detect(file_path):
    image_bgr = cv2.imread(file_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    results = detector.detect_faces(image_rgb)
    if not results:
        label1.configure(foreground="#011638", text="No face detected.")
        show_image(image_rgb)
        return

    # Use the first detected face
    x, y, w, h = results[0]['box']
    x, y = abs(x), abs(y)
    face = image_rgb[y:y + h, x:x + w]

    try:
        face_resized = cv2.resize(face, (300, 300))
    except Exception as e:
        label1.configure(foreground="#011638", text="Error processing face region.")
        return

    img_array = face_resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction)
    pred_label = CLASS_NAMES[pred_index]
    confidence = prediction[0][pred_index] * 100

    label1.configure(foreground="#011638", text=f"Prediction: {pred_label} ({confidence:.2f}%)")

    # Draw bounding box around detected face
    cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

    show_image(image_rgb)

def show_image(img_rgb):
    img_pil = Image.fromarray(img_rgb)
    img_pil.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(img_pil)
    sign_image.configure(image=img_tk)
    sign_image.image = img_tk

def show_Detect_button(file_path):
    detect_b = Button(top, text="Detect Condition", command=lambda: Detect(file_path), padx=10, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)

def upload_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    try:
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_Detect_button(file_path)
    except Exception as e:
        label1.configure(text="Unsupported file format")

# GUI layout
upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=50)

sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')

heading = Label(top, text='Down Syndrome Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

top.mainloop()
