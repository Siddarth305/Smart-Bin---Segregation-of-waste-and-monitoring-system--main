import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the model
model_path = 'E:/Smartbin_garbage/waste_classifier/waste_classifier.keras'
model = tf.keras.models.load_model(model_path)

# Function to predict the waste type
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Ensure the target size matches the training size
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])  # Get the index of the highest prediction
    return predicted_class

# Example usage
test_image_path = 'E:/Smartbin_garbage/processed_images/plastic/plastic_001.jpg'  # Change this to your test image
predicted_label = predict_image(test_image_path)
print(f'The predicted waste type is: {predicted_label}')
