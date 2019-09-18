import base64
import io
import os

import cv2
import numpy as np
import tensorflow as tf
from imageio import imread
from sanic import Sanic
from sanic import response
from sanic_jinja2 import SanicJinja2  # pip install sanic_jinja2

app = Sanic()
jinja = SanicJinja2(app)

model = None  # Deep Learning model

dir_path = os.path.dirname(os.path.realpath(__file__))
app.config.filefolder = dir_path + "/uploads"
# Serves files from the static folder to the URL /static
app.static('/js', './js')


@app.route('/')
@jinja.template('index.html')  # decorator method is static method
async def index(request):
    return


@app.route("/upload", methods=['POST'])
async def on_file_upload(request):
    imgdata = base64.b64decode(request.form['imgBase64'][0].split(",")[1])
    raw_img = imread(io.BytesIO(imgdata))
    gray_image = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(gray_image, (28, 28))

    # reconstruct image as an numpy array
    input = np.asarray([image])

    input = input.reshape(input.shape[0], 28, 28, 1)
    float_input = input.astype('float32') / 255

    predictions = model.predict(float_input)
    '''
    np.set_printoptions(precision=3, suppress=True)
    print(predictions)
    '''

    return response.json(str(np.argmax(predictions)))


if __name__ == "__main__":
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

    app.run(host="0.0.0.0", port=8000)
