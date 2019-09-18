# Code từ site này https://github.com/minsuk-heo/python_tutorial/blob/master/data_science/nlp/word2vec_tensorflow.ipynb
import pandas as pd
corpus = ['king is a strong man',
          'queen is a wise woman',
          'boy is a young man',
          'girl is a young woman',
          'prince is a young king',
          'princess is a young queen',
          'man is strong',
          'woman is pretty',
          'prince is a boy will be king',
          'princess is a girl will be queen']


def remove_stop_words(corpus):
    stop_words = ['is', 'a', 'will', 'be']
    results = []
    for text in corpus:
        tmp = text.split(' ')
        for stop_word in stop_words:
            if stop_word in tmp:
                tmp.remove(stop_word)
        results.append(" ".join(tmp))

    return results


corpus = remove_stop_words(corpus)


def extract_set_word(corpus):
    words = []
    for text in corpus:
        for word in text.split(' '):
            words.append(word)

    return set(words)


words = extract_set_word(corpus)

word2int = {}

for i, word in enumerate(words):
    word2int[word] = i

print(word2int)
sentences = []  # mảng chưa các mảng từ tạo nên một sentence
for sentence in corpus:
    sentences.append(sentence.split())

WINDOW_SIZE = 2

data = []
for sentence in sentences:  #
    for idx, word in enumerate(sentence):
        # Tìm ra các từ bên trái và bên phải từ word, nếu nó khác thì cho vào dữ liệu
        for neighbor in sentence[max(idx - WINDOW_SIZE, 0): min(idx + WINDOW_SIZE, len(sentence)) + 1]:
            if neighbor != word:
                data.append([word, neighbor])



df = pd.DataFrame(data, columns=['input', 'label'])
print(df.head(10))
