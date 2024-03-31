import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the saved model
model = tf.keras.models.load_model('gpt_mini.h5')

# Compile the model manually
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())
# Function to make predictions
def predict_digit(image):
    # Preprocess the image if necessary
    # Perform prediction
    prediction = model.predict(image)
    return prediction

# Example usage
# image = preprocess_image(image)
# prediction = predict_digit(image)
