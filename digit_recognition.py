import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Load your saved model
model = load_model('weights.h5')

def load_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"The image file '{img_path}' does not exist.")
    img = load_img(img_path, color_mode='grayscale', target_size=(28, 28))
    img = img_to_array(img)
    img = img.astype('float32') / 255
    img = 1 - img  # Invert image if part of training pipeline
    img = img.reshape(1, 28, 28, 1)  # Include channel dimension
    return img

# Load image
img = load_image('ab.png')

# Verify shape and pixel range
print("Image shape: ", img.shape)
print("Pixel range: ", img.min(), "-", img.max())

# Plot image
plt.imshow(img[0, :, :, 0], cmap='gray')
plt.title("Preprocessed Image")
plt.show()

# Predict using the model
predictions = model.predict(img)

# Get the predicted digit
predicted_digit = np.argmax(predictions[0])

print("Predicted digit:", predicted_digit)

# Print prediction probabilities
print("\nPrediction Probabilities:")
for idx, prob in enumerate(predictions[0]):
    print(f"Class {idx}: {prob:.8f}")
