from flask import Flask, request, jsonify, render_template
from tensorflow import keras
import numpy as np
from PIL import Image # For image processing
import io # To handle image bytes
import os # For checking if model file exists

app = Flask(__name__)

# --- Model Loading ---
model = None
model_path = 'mnist_cnn_model.h5'
if not os.path.exists(model_path):
    print(f"ERROR: Model file '{model_path}' not found! Please ensure it's in the same directory as app.py")
else:
    try:
        model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
        model.summary()
    except Exception as e:
        print(f"ERROR: Failed to load model '{model_path}': {e}")
        # raise e # Commented out unless you specifically want the app to crash on model load

# --- Route Definitions ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST']) # This line is correct in your code
def predict():
    print("Received prediction request.") # Debug print

    if model is None:
        print("Error: Model is not loaded.") # Debug print
        return jsonify({'error': 'Model not loaded on server. Please check server logs.'}), 500

    if 'file' not in request.files:
        print("Error: No 'file' part in request.") # Debug print
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        print("Error: No selected file in request.") # Debug print
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            print(f"Processing file: {file.filename}")
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            print(f"Original image mode: {img.mode}, size: {img.size}")

            img = img.convert('L')
            print(f"After grayscale conversion, image mode: {img.mode}")

            img = img.resize((28, 28))
            print(f"After resize, image size: {img.size}")

            img_array = np.array(img)
            print(f"NumPy array shape after conversion: {img_array.shape}")
            print(f"NumPy array dtype: {img_array.dtype}")
            print(f"Min/Max pixel value before normalization: {img_array.min()}/{img_array.max()}")

            img_array = 255 - img_array # Color inversion
            print(f"Min/Max pixel value AFTER INVERSION: {img_array.min()}/{img_array.max()}")

            img_array = img_array / 255.0
            print(f"Min/Max pixel value after normalization: {img_array.min()}/{img_array.max()}")

            img_array = img_array.reshape(1, 28, 28, 1)
            print(f"Final input shape for model: {img_array.shape}")

            prediction = model.predict(img_array)
            print(f"Raw prediction output: {prediction}")

            predicted_digit = np.argmax(prediction)
            print(f"Predicted digit: {predicted_digit}")

            return jsonify({'digit': int(predicted_digit)})

        except Exception as e:
            print(f"An error occurred during image processing or prediction: {e}")
            return jsonify({'error': str(e)}), 500

# --- NEW DEBUGGING CODE ---
# This function will print Flask's URL map on startup
with app.test_request_context():
    print("\n--- Flask URL Map ---")
    for rule in app.url_map.iter_rules():
        print(f"Endpoint: {rule.endpoint}, Path: {rule.rule}, Methods: {rule.methods}")
    print("-------------------\n")
# --- END NEW DEBUGGING CODE ---

if __name__ == '__main__':
    app.run(debug=True)