from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow import keras
from keras.layers.embeddings import Embedding

# define documents
docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']
# define class labels
labels = array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
print(f"vocab_size {vocab_size}")

# integer encode the documents
#vocab_size = 50
# pad documents to a max length of 4 words
max_length = 4


def process_docs(raw_docs):
    encoded_docs = [one_hot(d, vocab_size) for d in raw_docs]
    print(encoded_docs)

    padded = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    return padded


padded_docs = process_docs(docs)
print(padded_docs)

# define the model
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 8, input_length=max_length))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# summarize the model
print(model.summary())

# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=2)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy * 100))

test_docs = ['Good work', 'Great effort', 'Excellent!', 'not good', 'poor']

test_padded_docs = process_docs(test_docs)
result = model.predict(test_padded_docs)
print(result)
