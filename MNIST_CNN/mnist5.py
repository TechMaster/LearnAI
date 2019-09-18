import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

test_folder = str(Path(__file__).parent.parent / 'MNIST_TEST/')
test_images = []
for file in os.listdir(test_folder):
    if file.endswith(".png") or file.endswith(".jpg"):
        img_path = os.path.join(test_folder, file)
        print(file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        test_images.append(img)


input = np.asarray(test_images)

input = input.reshape(input.shape[0], 28, 28, 1)
test_data = input.astype('float32') / 255

predictions = model.predict(test_data)
np.set_printoptions(precision=3, suppress=True)
print(predictions)
