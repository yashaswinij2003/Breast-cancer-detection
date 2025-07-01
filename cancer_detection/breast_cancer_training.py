# training the model
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split

# 🔹 Define dataset path
image_directory = "C:/Users/yasha/OneDrive/Documents/python_practice/majorproject/dataset_folder"

# 🔹 Load dataset function
def load_data(directory):
    images = []
    labels = []

    # Verify dataset exists
    if not os.path.exists(directory):
        raise ValueError(f"❌ Dataset path '{directory}' does not exist!")

    for category in ["Cancer", "Non-Cancer"]:
        category_path = os.path.join(directory, category)

        if not os.path.exists(category_path):
            print(f"⚠ Skipping {category}: Path does not exist.")
            continue

        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)

            try:
                img = tf.keras.preprocessing.image.load_img(file_path, target_size=(224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                images.append(img_array)

                labels.append(1 if category.lower() == "cancer" else 0)

            except Exception as e:
                print(f"⚠ Skipping {file_path}: {e}")

    return np.array(images), np.array(labels)

# 🔹 Load images & labels
images, labels = load_data(image_directory)
print(f"🔹 Loaded: {len(images)} images, {len(labels)} labels.")

# 🔹 Ensure dataset isn't empty before training
if len(images) == 0 or len(labels) == 0:
    raise ValueError("❌ Dataset is empty! Check dataset loading.")

# 🔹 Split Data
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
print(f"✅ Data split: {len(X_train)} training samples, {len(X_val)} validation samples.")

# 🔹 Define Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 🔹 Compile Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 🔹 Train Model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# 🔹 Save Model for Future Use
model.save("breast_cancer_classifier.keras")  # Save in Keras format

print("🎉 Training complete! Model saved successfully.")
print(f"🔹 Total images loaded: {len(images)}")
print(f"🔹 Total labels loaded: {len(labels)}")






