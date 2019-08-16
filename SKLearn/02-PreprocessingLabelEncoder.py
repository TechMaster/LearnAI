from sklearn import preprocessing

le = preprocessing.LabelEncoder()
print(le.fit(["paris", "paris", "tokyo", "amsterdam", "hà nội", "sài gòn"]))

list(le.classes_)

print(le.transform(["tokyo", "tokyo", "paris", "sài gòn"]))

print(list(le.inverse_transform([2, 2, 1, 3, 0])))

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html