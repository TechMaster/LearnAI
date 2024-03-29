import tensorflow as tf
from tensorflow import keras



imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))


# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


#Paddinig data
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

#print(len(train_data[0]), len(train_data[1]))

print(train_data[0])
print(train_labels[0], decode_review(train_data[0]))
print(train_labels[6], decode_review(train_data[6]))
print(train_labels[8], decode_review(train_data[8]))
print(train_labels[5000], decode_review(train_data[5000]))



# Find positive review in train_labels
count = 0
for index, label in enumerate(train_labels):
    if label == 1:
        print(index)
        count += 1
        if count > 10:
            break

