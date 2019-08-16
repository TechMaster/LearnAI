from sklearn.preprocessing import Binarizer

X = [[1, 3, 2, 5, 0, 1]]
transformer = Binarizer(threshold=2)
print(transformer)

print(transformer.transform(X))
