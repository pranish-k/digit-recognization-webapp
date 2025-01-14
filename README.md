# Digit Recognition Web Application

This project provides a web application for digit recognition using a neural network trained on the MNIST dataset. The web app allows users to upload images of handwritten digits, which are then processed and classified by a trained neural network model.

## Features
- Upload an image of a handwritten digit (28x28 grayscale).
- The model predicts the digit in the image.
- Provides an interactive web interface for ease of use.

## Technologies Used
- Flask for the web application.
- TensorFlow/Keras for building and training the neural network.
- HTML for the frontend.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create a Virtual Environment
```bash
python3 -m venv env
source env/bin/activate   # For Linux/MacOS
env\Scripts\activate    # For Windows
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

Make sure the following packages are installed:
- Flask
- TensorFlow
- numpy

### 4. Train the Neural Network (Optional)
If you want to retrain the model, use the provided Jupyter notebook `train_mnist_model.ipynb`:

1. Load and preprocess the MNIST dataset.
2. Train a neural network with two hidden layers of 16 neurons each.
3. Save the model weights to a file named `weights.h5`.

### Example Code for Training
```python
import tensorflow as tf

# Load MNIST data
mnist = tf.keras.datasets.mnist
mnist_data = mnist.load_data()
train, train_label = mnist_data[0]
test, test_label = mnist_data[1]

# Normalize the data
train = train.astype('float32') / 255
test = test.astype('float32') / 255

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train, train_label, epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(test, test_label, verbose=2)
print('\nTest accuracy:', test_acc)

# Save the model
model.save('weights.h5')
```

### 5. Run the Flask Application
```bash
flask run
```

By default, the application will be available at `http://127.0.0.1:5000/`.

### 6. Using the Application
1. Open a web browser and navigate to `http://127.0.0.1:5000/`.
2. Upload an image of a handwritten digit.
3. The application will display the predicted digit.

---

## File Structure
```
.
├── app.py                     # Flask application
├── weights.h5                 # Pre-trained model weights
├── train_mnist_model.ipynb    # Jupyter notebook for training the model
└── templates                  # HTML templates (if using Flask templates)
```

## Requirements
- Python 3.8+
- Flask
- TensorFlow
- numpy

## Troubleshooting
1. **Error loading model:** Ensure `weights.h5` is present in the project directory.
2. **No file uploaded error:** Ensure you select an image before submitting.
3. **Prediction errors:** Ensure the uploaded image is a 28x28 grayscale image. Use image preprocessing if necessary.

## Future Improvements
- Add support for colored images.
- Improve the model's accuracy by using more complex architectures.
- Enhance the user interface.
- Add validation checks for uploaded files.

## Acknowledgments
- The MNIST dataset is publicly available and widely used for digit recognition tasks.
- This project was inspired by real-world applications of machine learning in computer vision.

## License
This project is licensed under the MIT License.

