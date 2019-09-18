import string

import tensorflow as tf
from tensorflow import keras
from keras.models import Model

class SentimentAnalysis:
    def __init__(self):
        self.model = SentimentAnalysis.load_model()
        self.word_index = SentimentAnalysis.load_imdb_word_index()

    # Load model đã được huấn luyện ở bước trước
    @staticmethod
    def load_model():
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
        return model

    # Load dictionary ánh xạ word sang index dạng integer
    @staticmethod
    def load_imdb_word_index():
        imdb = keras.datasets.imdb

        # A dictionary mapping words to an integer index
        word_index = imdb.get_word_index()

        # The first indices are reserved
        word_index = {k: (v + 3) for k, v in word_index.items()}
        word_index["<PAD>"] = 0
        word_index["<START>"] = 1
        word_index["<UNK>"] = 2  # unknown
        word_index["<UNUSED>"] = 3
        return word_index

    # Chuyển film review từ text sang mảng các số index
    def convert_text_to_index(self, text):
        lowertext = text.lower()
        translator = str.maketrans('', '', string.punctuation)  # Loại bỏ các
        lowertext = lowertext.translate(translator)
        split_text = lowertext.split(' ')
        encoded_text = []
        for word in split_text:
            index = self.word_index.get(word)
            if index is None or index >= 10000: # Không tồn tại hoặc lớn hơn 10000 gán là <UNK>
                index = 2

            # index = index if index < 10000 else 2
            encoded_text.append(index)
        return encoded_text

    def analyze_sentiment(self, raw_review):
        encoded_text = self.convert_text_to_index(raw_review)

        encoded_texts = [encoded_text]

        input_data = keras.preprocessing.sequence.pad_sequences(encoded_texts,
                                                                value=self.word_index["<PAD>"],
                                                                padding='post',
                                                                maxlen=256)

        result = self.model.predict(input_data)
        return result[0]

    def debug_hidden_layer_out(self, raw_review):
        encoded_text = self.convert_text_to_index(raw_review)

        encoded_texts = [encoded_text]

        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=self.model.get_layer("embedding").output)
        intermediate_output = intermediate_layer_model.predict(encoded_texts)
        return intermediate_output