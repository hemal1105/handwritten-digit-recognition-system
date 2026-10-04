from flask import Flask, request, jsonify, render_template
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps
import io
import os
from scipy.ndimage import center_of_mass

app = Flask(__name__)

model = None
model_path = 'mnist_cnn_model.h5'
if os.path.exists(model_path):
    try:
        model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"ERROR loading model: {e}")

def preprocess_mnist_style(pil_img):
    # 1. Convert to grayscale
    img = pil_img.convert('L')
    
    # 2. Convert to numpy array
    img_array = np.array(img)
    
    # Check if image background is mostly white (e.g. uploaded images vs canvas)
    # MNIST expects bright/white digits (high pixel value) on dark/black backgrounds (0 pixel value).
    if np.mean(img_array) > 127:
        img_array = 255 - img_array  # Invert only if background is light

    # 3. Crop bounding box around the drawn digit
    coords = np.argwhere(img_array > 30)
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Crop to bounding box
        cropped = img_array[y_min:y_max+1, x_min:x_max+1]
        
        # Resize cropped digit to fit inside a 20x20 box while preserving aspect ratio
        h, w = cropped.shape
        if h > w:
            new_h = 20
            new_w = max(1, int(round(w * (20.0 / h))))
        else:
            new_w = 20
            new_h = max(1, int(round(h * (20.0 / w))))
            
        digit_img = Image.fromarray(cropped).resize((new_w, new_h), Image.Resampling.LANCZOS)
        digit_arr = np.array(digit_img)
        
        # 4. Pad image to 28x28 and center using center of mass
        padded = np.zeros((28, 28), dtype=np.float32)
        pad_y = (28 - new_h) // 2
        pad_x = (28 - new_w) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = digit_arr
        
        # Fine-tune position via center-of-mass shift (matching MNIST benchmark pipeline)
        cy, cx = center_of_mass(padded)
        if not np.isnan(cy) and not np.isnan(cx):
            shift_y = int(round(13.5 - cy))
            shift_x = int(round(13.5 - cx))
            padded = np.roll(padded, shift_y, axis=0)
            padded = np.roll(padded, shift_x, axis=1)
            
        img_array = padded
    else:
        # Fallback if canvas is empty
        img_array = np.zeros((28, 28), dtype=np.float32)

    # 5. Normalize pixel values to [0, 1]
    img_array = img_array / 255.0
    return img_array.reshape(1, 28, 28, 1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # Preprocess image to match MNIST distribution precisely
        processed_input = preprocess_mnist_style(img)
        
        # Perform prediction
        prediction = model.predict(processed_input)
        predicted_digit = np.argmax(prediction)
        
        return jsonify({'digit': int(predicted_digit)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)