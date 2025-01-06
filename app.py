from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Ensure uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load your pre-trained model
model = load_model('my_model.h5')  # Replace with your .h5 model file
classes = ['Almond', 'Cashew', 'Peanuts','Pistachio','Walnuts','Wild Almond','Not a Nut']  # Replace with your class names

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the file temporarily
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Preprocess the image
    img = load_img(file_path, target_size=(224, 224))  # Adjust size for your model
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = classes[np.argmax(predictions)]

    # Clean up
    os.remove(file_path)

    return jsonify({'predicted_class': predicted_class})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
