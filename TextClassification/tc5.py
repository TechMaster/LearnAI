import string

import tensorflow as tf
from tensorflow import keras

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

# -----------------
imdb = keras.datasets.imdb

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3


def convert_text_to_index(text):
    lowertext = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    lowertext = lowertext.translate(translator)
    print(lowertext)
    split_text = lowertext.split(' ')
    encoded_text = []
    for word in split_text:
        index = word_index.get(word)
        if index is None or index >= 10000:
            index = 2

        # index = index if index < 10000 else 2
        encoded_text.append(index)
    return encoded_text


encoded_text = convert_text_to_index(
    "I'm not a professional reviewer. For me, I enjoyed the acting in the film. I didn't see point to the story, didn't feel like benefited from seeing this film and felt it difficult to see through to the end. I wouldn't go so far as to say it's a bad movie, I think the sound, the cinematography was all good and as said the acting. But the story, I guess I missed the point here.")
print(encoded_text)

encoded_texts = [encoded_text]

input_data = keras.preprocessing.sequence.pad_sequences(encoded_texts,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

result = model.predict(input_data)
print(result[0])
