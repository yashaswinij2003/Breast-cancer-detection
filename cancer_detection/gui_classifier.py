#gui
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# 🔹 Load the trained model
model = tf.keras.models.load_model("breast_cancer_classifier.keras")  # Use .h5 if needed

# 🔹 List of symptom questions
symptoms_questions = [
    "Have you noticed a lump in your breast?",
    "Has your breast shape changed recently?",
    "Do you experience nipple discharge?",
    "Do you feel pain in your breast?",
    "Have you noticed any skin dimpling?",
    "Is there redness or swelling in your breast?",
    "Do you have itchy or scaly skin?",
    "Has your nipple appearance changed?",
    "Have you felt swollen lymph nodes?",
    "Are you experiencing persistent fatigue?",
    "Have you had unexplained weight loss?",
    "Do you feel thickening in the breast?",
    "Have you felt a hard knot near your breast?",
    "Do you have pain in your underarm?",
    "Has your nipple turned inward recently?"
]

# 🔹 Extracted symptom names (for clean summary display)
symptom_names = [
    "Lump in breast",
    "Change in breast shape",
    "Nipple discharge",
    "Pain in breast",
    "Skin dimpling",
    "Redness or swelling",
    "Itchy or scaly skin",
    "Change in nipple appearance",
    "Swollen lymph nodes",
    "Persistent fatigue",
    "Unexplained weight loss",
    "Breast thickening",
    "Hard knot near breast",
    "Pain in underarm",
    "Nipple inversion"
]

# 🔹 Store user responses
user_responses = []
current_question_index = 0  # Tracks current symptom being asked

# 🔹 Function to run image prediction
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)
    return "Cancer" if prediction[0][0] > 0.5 else "Non-Cancer"

# 🔹 Function to start asking symptom questions after image prediction
def ask_symptoms():
    global current_question_index
    current_question_index = 0
    user_responses.clear()  # Reset responses
    question_label.config(text=symptoms_questions[current_question_index])

    yes_button.config(state="normal")
    no_button.config(state="normal")

# 🔹 Function to store Yes/No responses and move to the next question
def store_response(response):
    global current_question_index
    if response:
        user_responses.append(symptom_names[current_question_index])  # Store only symptom name

    current_question_index += 1

    if current_question_index < len(symptoms_questions):
        question_label.config(text=symptoms_questions[current_question_index])
    else:
        yes_button.config(state="disabled")
        no_button.config(state="disabled")
        show_summary()

# 🔹 Display summary after answering all questions
def show_summary():
    summary = "\n".join(user_responses) if user_responses else "No symptoms selected."
    messagebox.showinfo("Symptoms Detected", summary)

# 🔹 Function to open and predict image
def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((224, 224))
        img = ImageTk.PhotoImage(img)

        image_label.config(image=img)
        image_label.image = img

        result = predict_image(file_path)
        result_label.config(text=f"Prediction: {result}")

        # After prediction, start asking symptoms
        ask_symptoms()

# 🔹 Tkinter Window
root = tk.Tk()
root.title("Breast Cancer Classifier")
root.geometry("500x500")  # Centered layout

# 🔹 Main Frame (Centers all elements)
main_frame = tk.Frame(root)
main_frame.pack(expand=True)

# 🔹 Open Image Button
open_button = tk.Button(main_frame, text="Upload Image", command=open_file, font=("Arial", 12))
open_button.pack(pady=10)

image_label = tk.Label(main_frame)
image_label.pack()

result_label = tk.Label(main_frame, text="Prediction: Waiting...", font=("Arial", 14))
result_label.pack(pady=10)

# 🔹 Question Label for Symptoms (Initially Empty)
question_label = tk.Label(main_frame, text="", font=("Arial", 12))
question_label.pack()

# 🔹 Yes/No Buttons for Symptoms (Initially Disabled)
button_frame = tk.Frame(main_frame)
button_frame.pack(pady=10)

yes_button = tk.Button(button_frame, text="Yes", command=lambda: store_response(True), state="disabled", font=("Arial", 12))
yes_button.pack(side="left", padx=10)

no_button = tk.Button(button_frame, text="No", command=lambda: store_response(False), state="disabled", font=("Arial", 12))
no_button.pack(side="right", padx=10)

# 🔹 Run GUI
root.mainloop()