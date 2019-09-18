import tensorflow as tf
from tensorflow import keras


imdb = keras.datasets.imdb

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3


import string
def convert_text_to_index(text):
    lowertext = text.lower()
    translator = str.maketrans('','',string.punctuation)
    lowertext = lowertext.translate(translator)
    print(lowertext)
    split_text = lowertext.split(' ')
    encoded_text = []
    for word in split_text:
        index = word_index.get(word)
        if index is None or index >= 10000:
            index = 2
        encoded_text.append(index)
    return encoded_text

encoded_text = convert_text_to_index("Bite the hand that feeds you is the socialist motto of this film. While deeming capitalism and comparing US capitalism to totalitarianism capitalism, this film is trying to fit the narrative to socialize the US while also telling those of small countries or small island groups to look away from capitalism and stay within your tradition. I do agree with the latter, but it is nearly a fantasy that is ignorant to the expansion and evolution on an 'intelligent' or industrialized civilisation. I am not saying that I disagree with all of CG's messages, but only stating an observation of the ill ideology popularized fad on the internet. It's hard to watch CG talk about socialising the island while he enjoys the fruits of modern technology, parading around all high and mighty on his moral high ground blessed by the people around him. It's hard to like this movie when he demonises the US while comparing it to a full work week controlled by a totalitarian dictator of which we do not have.")
encoded_texts = [encoded_text]

input_data = keras.preprocessing.sequence.pad_sequences(encoded_texts,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

print(input_data)