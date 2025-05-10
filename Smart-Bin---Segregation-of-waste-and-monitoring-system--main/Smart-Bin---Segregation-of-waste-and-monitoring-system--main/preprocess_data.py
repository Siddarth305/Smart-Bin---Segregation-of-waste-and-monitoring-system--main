import os
import cv2

# Path to your dataset
dataset_path = 'E:/Smartbin_garbage/processed_images/'

# Loop through each category
for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    
    # Loop through each image in the category
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        
        # Read and resize the image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (150, 150))  # Resize to match model input
        
        # Save the preprocessed image (optional)
        cv2.imwrite(img_path, img)
