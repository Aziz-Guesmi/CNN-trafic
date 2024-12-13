import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf  

# Constants
IMG_SIZE = (416, 416)
names = [
    'Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100', 'Speed Limit 110',
    'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30', 'Speed Limit 40',
    'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70', 'Speed Limit 80',
    'Speed Limit 90', 'Stop'
]

# Model for darknet points detection
yolo_model_path = "bestmoretrainning.pt"  
model_yolo = YOLO(yolo_model_path)

cnn_model_path = "cnn_valid.h5"  
model_cnn = tf.keras.models.load_model(cnn_model_path)

# Function to preprocess images for CNN
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image) / 255.0 
    image = np.expand_dims(image, axis=0)  
    return image

st.title("CNN Traffic Sign Detection and Classification")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Display the original image once
    st.image(image, caption="Uploaded Image with Predictions", use_column_width=True)

    st.write("Processing ...")

    temp_image_path = "temp_image.jpg"
    image.save(temp_image_path)

    results = model_yolo.predict(temp_image_path)

    detections = []
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", size=14)  
    except IOError:
        font = ImageFont.load_default() 

    for result in results:
        for detection in result.boxes.data:
            x_min, y_min, x_max, y_max, score, class_id = detection.tolist()
            if int(class_id) in range(len(names)):  # Filter only panel classes
                detections.append((x_min, y_min, x_max, y_max, int(class_id)))

                # Draw bounding box
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

                # Classify the traffic sign using the CNN model
                cropped_image = image.crop((x_min, y_min, x_max, y_max))
                processed_image = preprocess_image(cropped_image, IMG_SIZE)
                predictions = model_cnn.predict(processed_image)
                predicted_class_index = np.argmax(predictions)
                predicted_class_name = names[predicted_class_index]

                text_y_position = max(y_min - 20, 0)  # Move 20 pixels above, ensuring it doesn't go out of the image

                draw.text((x_min, text_y_position), predicted_class_name, fill="red", font=font)

    st.image(image, caption="Processed Image with Predictions", use_column_width=True)
