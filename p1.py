import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape images to include the channel dimension
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Build a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Save the model
model.save('garment_model.h5')

# Load the model
garment_model = load_model('garment_model.h5')

# Preprocess image function
def preprocess_image(image_path):
    # Load the image
    img = load_img(image_path, color_mode='grayscale', target_size=(28, 28))  # Resizing to the model's expected input size
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Adding batch dimension
    img = img / 255.0  # Normalize pixel values
    return img
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File '{image_path}' not found.")
    
    img = load_img(image_path, color_mode='grayscale', target_size=(28, 28))
    # Further preprocessing steps...
    return img 


# Predict details function
def predict_details(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)
    
    # Predict garment type
    garment_pred = garment_model.predict(img)
    garment_type = class_names[np.argmax(garment_pred)]
    
    return {
        'Garment Type': garment_type
    }

# Example usage
image_path = r'C:\Users\Bhavya Anand\OneDrive\Desktop\shrt1.jpg'
details = predict_details(image_path)
print(details)

# Display the image and prediction
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.title(f"Predicted: {details['Garment Type']}")
plt.show()