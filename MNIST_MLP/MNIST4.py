import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()


model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


(_, _), (test_images, test_labels) = mnist.load_data()
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
test_labels = to_categorical(test_labels)


score = model.evaluate(test_images, test_labels)

print(f"{model.metrics_names[1]} : {score[1] * 100}")

