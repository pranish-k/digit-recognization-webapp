from flask import Flask, request, render_template_string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
model = load_model('weights.h5')


def load_image(img_path):
    """Load and preprocess the image."""
    img = load_img(img_path, color_mode='grayscale', target_size=(28, 28))
    img = img_to_array(img)
    img = img.astype('float32') / 255
    img = 1 - img  # Invert image if part of training pipeline
    img = img.reshape(1, 28, 28, 1)  # Include channel dimension
    return img


@app.route('/', methods=['GET'])
def index():
    """Serve the HTML page for image upload."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Digit Recognition</title>
    </head>
    <body>
        <h1>Upload an Image for Digit Recognition</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*">
            <button type="submit">Predict</button>
        </form>
    </body>
    </html>
    '''


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    if 'file' not in request.files:
        return "No file uploaded.", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected.", 400

    try:
        file_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(file_path)

        # Preprocess the image
        img = load_image(file_path)

        # Predict using the model
        predictions = model.predict(img)
        predicted_digit = int(np.argmax(predictions[0]))

        # Return an HTML response with the result
        return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prediction Result</title>
        </head>
        <body>
            <h1>Prediction Result</h1>
            <p>I think the number in the image is <strong>{{ predicted_digit }}</strong>.</p>
            <a href="/">Upload another image</a>
        </body>
        </html>
        ''', predicted_digit=predicted_digit)
    except Exception as e:
        return str(e), 500


if __name__ == '__main__':
    app.run(debug=True)
