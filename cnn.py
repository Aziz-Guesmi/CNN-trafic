import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import os
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Constants
IMG_SIZE = (416, 416)
names = [
    'Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100', 'Speed Limit 110',
    'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30', 'Speed Limit 40',
    'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70', 'Speed Limit 80',
    'Speed Limit 90', 'Stop'
]

# Load models
yolo_model_path = "bestmoretrainning.pt"
model_yolo = YOLO(yolo_model_path)

cnn_model_path = "cnn_valid.h5"
model_cnn = tf.keras.models.load_model(cnn_model_path)

# Root dataset directory
dataset = "C:/Users/agues/cnn-projet/DataSet/"

# Train paths
train_images_path = os.path.join(dataset, 'train/images/')
train_labels_path = os.path.join(dataset, 'train/labels/')

# Validation paths
valid_images_path = os.path.join(dataset, 'valid/images/')
valid_labels_path = os.path.join(dataset, 'valid/labels/')

# Test paths
test_images_path = os.path.join(dataset, 'test/images/')
test_labels_path = os.path.join(dataset, 'test/labels/')




# Parameters
IMG_SIZE = (416, 416)  # Target image size
NUM_CLASSES = 15       # Number of classes

class SignDataGenerator(Sequence):
    def __init__(self, image_dir, label_dir, batch_size,augmentation=False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.augmentation = augmentation
        # Verify valid pairs where the image exists
        self.image_files, self.label_files = self.filter_valid_pairs()

    def filter_valid_pairs(self):
      """Filter out label-image pairs where the image does not exist and shuffle the pairs."""
      valid_image_files = []
      valid_label_files = []

      label_files = [f for f in os.listdir(self.label_dir) if f.endswith('.txt')]
      for label_file in label_files:
          image_file = label_file.replace('.txt', '.jpg')  # Match image name
          image_path = os.path.join(self.image_dir, image_file)
          if os.path.exists(image_path):  # Check if the image exists
              valid_image_files.append(image_file)
              valid_label_files.append(label_file)
          else:
              print(f"Image not found for label: {label_file}")  # Log missing image

      # Shuffle both lists while maintaining correspondence
      #paired_files = list(zip(valid_image_files, valid_label_files))
      #random.shuffle(paired_files)  # Shuffle the list of tuples
      #valid_image_files, valid_label_files = zip(*paired_files)  # Unzip back into two lists

      return valid_image_files, valid_label_files

    def __len__(self):
        return int(np.ceil(len(self.image_files) / self.batch_size))

    def parse_label(self, label_path):
        bounding_boxes = []
        with open(label_path, 'r') as f:
            lines = f.readlines()  # Read all lines in the label file

        for line in lines:
            parts = line.strip().split()  # Split line into parts
            if parts:
                class_id = int(parts[0])  # The class ID is the first element
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                bounding_boxes.append((class_id, x_center, y_center, width, height))
            #else:
             #   class_id = 0  # Default to 0 if line is empty

        return bounding_boxes

    from PIL import Image, ImageEnhance

    def crop_image(self, image, bounding_boxes):
        cropped_images = []
        img_width, img_height = image.size

        for class_id, x_center, y_center, width, height in bounding_boxes:
            x_min = int((x_center - width / 2) * img_width)
            y_min = int((y_center - height / 2) * img_height)
            x_max = int((x_center + width / 2) * img_width)
            y_max = int((y_center + height / 2) * img_height)

            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            cropped_image = cropped_image.resize(IMG_SIZE)  
            cropped_images.append(img_to_array(cropped_image) / 255.0)  

            if self.augmentation:
                # Rotate augmentation
                for angle in [30, 60, 90, 120, 160 ,180,210,220,250, 270]:
                    rotated_image = cropped_image.rotate(angle)
                    cropped_images.append(img_to_array(rotated_image) / 255.0)

              

        return np.array(cropped_images)



    def get_all_labels(self):
        all_labels = []
        for image_file in self.image_files:
            label_file = image_file.replace('.jpg', '.txt')
            label_path = os.path.join(self.label_dir, label_file)
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        class_id = int(line.split()[0])
                        all_labels.append(class_id)
        return np.array(all_labels)


    def __getitem__(self, index):
        # Get batch indices
        batch_images = self.image_files[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.label_files[index * self.batch_size:(index + 1) * self.batch_size]

        # Load data
        images = []
        labels = []
        for img_name, label_name in zip(batch_images, batch_labels):
            # Load and preprocess the image
            img_path = os.path.join(self.image_dir, img_name)
            image = load_img(img_path)


            # Parse the label and get bounding box coordinates
            label_path = os.path.join(self.label_dir, label_name)
            bounding_boxes = self.parse_label(label_path)

            # Crop the image based on the bounding boxes
            cropped_images = self.crop_image(image, bounding_boxes)
            images.extend(cropped_images)  # Add cropped images to the list

            # Create one-hot labels for each cropped image
            for class_id, _, _, _, _ in bounding_boxes:
                label_one_hot = np.zeros(NUM_CLASSES)  # Create a zero vector of length NUM_CLASSES
                label_one_hot[class_id] = 1  # Set the index corresponding to the class_id to 1
                labels.append(label_one_hot)

        return np.array(images), np.array(labels)





def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def plot_confusion_matrix(conf_matrix, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)

# Streamlit layout
st.title("Traffic Sign Detection and Analysis")

# Sidebar for navigation
st.sidebar.header("Navigation")
choice = st.sidebar.radio(
    "Choose a feature to explore:",
    options=["Traffic Sign Detection", "Confusion Matrix Dashboard"],
    index=0,
    help="Select a feature to explore its functionality."
)

if choice == "Traffic Sign Detection":
    st.header("Traffic Sign Detection and Classification")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Processing with YOLO
        st.write("Processing with YOLO...")
        results = model_yolo.predict(image)

        detections = []
        draw = ImageDraw.Draw(image)

        # Load a font
        try:
            font = ImageFont.truetype("arial.ttf", size=14)
        except IOError:
            font = ImageFont.load_default()

        # Detect and classify objects
        for result in results:
            for detection in result.boxes.data:
                x_min, y_min, x_max, y_max, score, class_id = detection.tolist()
                if int(class_id) in range(len(names)):
                    detections.append((x_min, y_min, x_max, y_max, int(class_id)))

                    # Draw bounding box
                    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

                    # Classify the traffic sign using CNN
                    cropped_image = image.crop((x_min, y_min, x_max, y_max))
                    processed_image = preprocess_image(cropped_image, IMG_SIZE)
                    predictions = model_cnn.predict(processed_image)
                    predicted_class_index = np.argmax(predictions)
                    predicted_class_name = names[predicted_class_index]

                    # Annotate image
                    text_y_position = max(y_min - 20, 0)
                    draw.text((x_min, text_y_position), predicted_class_name, fill="red", font=font)

        # Display the updated image with annotations
        st.image(image, caption="Processed Image with Predictions", use_column_width=True)

elif choice == "Confusion Matrix Dashboard":
    st.header("Confusion Matrix Dashboard")

    # Instantiate data generators
    train_gen = SignDataGenerator(train_images_path, train_labels_path, batch_size=45)
    valid_gen = SignDataGenerator(valid_images_path, valid_labels_path, batch_size=45)
    test_gen = SignDataGenerator(test_images_path, test_labels_path, batch_size=45)
    st.write("### Generating Confusion Matrix...")
    test_predictions = model_cnn.predict(test_gen, verbose=1)  
    
    all_test_labels = test_gen.get_all_labels()

    # Convert predictions from probabilities to class labels
    predictions = np.argmax(test_predictions, axis=1)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_test_labels, predictions)  

    st.write("### Confusion Matrix")
    plot_confusion_matrix(conf_matrix, names)

    st.write("### Metrics")
    accuracy = accuracy_score(all_test_labels, predictions)
    st.write(f"Accuracy: {accuracy:.2f}")
