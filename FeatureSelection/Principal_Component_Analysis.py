# Feature Extraction with PCA
from pandas import read_csv
from sklearn.decomposition import PCA

# load data
filename = 'pima-indians-diabetes.data.csv'

data = read_csv(filename)
array = data.values
X = array[:, 0:8]
Y = array[:, 8]
# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print(f"Explained Variance: {fit.explained_variance_ratio_}")
print(fit.components_)


'''
StatQuest PCA
https://www.youtube.com/watch?v=HMOI_lkzW08
'''