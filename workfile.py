from sklearn import svm
from sklearn import datasets

# X = [[0, 0], [1, 1]]
# y = [0, 1]
# X is feature data
# y is class label

iris = datasets.load_iris()
classifier = svm.SVC()

classifier.fit(iris.data, iris.target)
print(iris.target_names)
print(classifier.predict([[100, 1, 1, 1]]))

target_names = ['awesome', 'rad', 'excellent']
feature_names = ['age', 'number of pets', 'GPA']
feature_data = [
    [10000, 2, 5.3],
    [-1, 3, 0.2],
    [30.1, 3, 4.2],
]

labels = [0, 1, 2]
classifier2 = svm.SVC()
classifier2.fit(feature_data, labels)
print(classifier2.predict([[45, 1, 3.2], [12, 4, 2.3]]))
