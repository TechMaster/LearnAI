# Feature Extraction with RFE
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# load data
filename = 'pima-indians-diabetes.data.csv'

data = read_csv(filename)
array = data.values
X = array[:, 0:8]
Y = array[:, 8]
# feature extraction
model = LogisticRegression(solver='liblinear')
rfe = RFE(model, 4)
fit = rfe.fit(X, Y)
print(f"Num Features: {fit.n_features_}")
print(f"Selected Features: {fit.support_}")
print(f"Feature Ranking: {fit.ranking_}")

new_features = []
for selected, feature in zip(fit.support_, data.columns):
    if selected:
        new_features.append(feature)

print(new_features)
