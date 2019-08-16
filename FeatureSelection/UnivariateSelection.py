# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# load data
filename = 'pima-indians-diabetes.data.csv'

data = read_csv(filename)
array = data.values
X = array[:, 0:8]
Y = array[:, 8]
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)

# summarize scores
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5, :])

# Lấy danh sách các feature phù nhợp nhất, có giá trị True
# học tập từ đây https://stackoverflow.com/questions/39839112/the-easiest-way-for-getting-feature-names-after-running-selectkbest-in-scikit-le
mask = test.get_support()

new_features = []  # The list of your K best features

for selected, feature in zip(mask, data.columns):
    if selected:
        new_features.append(feature)

print(new_features)
