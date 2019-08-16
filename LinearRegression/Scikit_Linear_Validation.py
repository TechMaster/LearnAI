import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('data_linear.csv').values

X = data[:, 0].reshape(-1, 1)
Y = data[:, 1].reshape(-1, 1)

test_size = 0.33
seed = 7

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = LinearRegression()
reg = model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)

print(f"Accuracy: {result * 100.0} %")
