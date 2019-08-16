from sklearn import preprocessing

enc = preprocessing.OneHotEncoder(sparse=False)
X = [
    ['male', 'from US', 'uses Safari'],
    ['female', 'from Europe', 'uses Firefox'],
    ['male', 'from Asia', 'uses Safari'],
    ['female', 'from US', 'uses IE'],
    ['male', 'from Europe', 'uses Chrome'],
    ['female', 'from US', 'uses Safari'],
    ['male', 'from Asia', 'uses Chrome'],
    ['female', 'from Asia', 'uses Firefox'],
]

enc.fit(X)

result = enc.transform([
    ['female', 'from US', 'uses Safari'],
    ['male', 'from US', 'uses Safari'],
    ['male', 'from Asia', 'uses Safari'],
])
print(result)
