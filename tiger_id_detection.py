import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load the trained model
model_path = r"C:\Users\thars\OneDrive\Desktop\project\data\Model\resnet50_model.h5"
model = load_model(model_path)

# Function to preprocess a single image
def preprocess_image(image_path, target_size=(200, 250)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image data
    return img_array

# Function to predict the tiger ID of an image
def predict_tiger_id(model, img_array):
    prediction = model.predict(img_array)
    tiger_id = np.argmax(prediction)
    return tiger_id

def main():
    st.title("Tiger ID Detection")

    st.write("Upload two images to check their tiger IDs.")

    uploaded_file1 = st.file_uploader("Choose the first image...", type=["jpg", "jpeg", "png"])
    uploaded_file2 = st.file_uploader("Choose the second image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file1 is not None and uploaded_file2 is not None:
        image1 = Image.open(uploaded_file1)
        image2 = Image.open(uploaded_file2)
        
        st.image(image1, caption='Uploaded Image 1', use_column_width=True)
        st.image(image2, caption='Uploaded Image 2', use_column_width=True)
        
        st.write("Classifying...")

        # Save the uploaded files temporarily
        image_path1 = "temp_image1.jpg"
        image_path2 = "temp_image2.jpg"
        image1.save(image_path1)
        image2.save(image_path2)

        # Preprocess images
        img_array1 = preprocess_image(image_path1)
        img_array2 = preprocess_image(image_path2)

        # Predict tiger IDs
        tiger_id1 = predict_tiger_id(model, img_array1)
        tiger_id2 = predict_tiger_id(model, img_array2)

        # Display tiger IDs
        st.write(f"Predicted Tiger ID for Image 1: {tiger_id1}")
        st.write(f"Predicted Tiger ID for Image 2: {tiger_id2}")

        if tiger_id1 == tiger_id2:
            st.write("These images belong to the same tiger ID.")
        else:
            st.write("These images belong to different tiger IDs.")
        
        # Remove the temporary files
        os.remove(image_path1)
        os.remove(image_path2)

if __name__ == "__main__":
    main()
