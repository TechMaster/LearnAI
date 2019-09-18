import itertools

import numpy as np


def get_data_sets():
    product_0_1 = list(itertools.product([0, 1], repeat=2))

    result = []
    for i in product_0_1:
        for j in product_0_1:
            result.append(np.vstack((i, j)))
    return result

    examples = get_data_sets()

    def training_set():
        while True:
            index = np.random.choice(len(examples))
            yield examples[index]

    def evaluation_set():
        while True:
            index = np.random.choice(len(examples))
            yield examples[index]

    return training_set, evaluation_set
