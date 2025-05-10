import cv2
import numpy as np
import tensorflow as tf

model_path = 'E:/Smartbin_garbage/waste_classifier/waste_classifier.keras'
model = tf.keras.models.load_model(model_path)

def predict_image(frame):
    img = cv2.resize(frame, (150, 150))  
    img_array = img / 255.0
    img_array = np.expand_dims(img_array, axis=0)  
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])  
    return predicted_class

class_labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'plastic'}

cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    roi = frame[height//4:height*3//4, width//4:width*3//4] 

    predicted_class = predict_image(roi)
    waste_type = class_labels.get(predicted_class, 'Unknown')

    cv2.rectangle(frame, (width//4, height//4), (width*3//4, height*3//4), (255, 0, 0), 2)
    cv2.putText(frame, f'Waste Type: {waste_type}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Waste Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
