from sklearn import tree
from matplotlib import pyplot as plt

X = [[0, 0], [1, 1]]
Y = [0, 1]
model = tree.DecisionTreeClassifier()
model = model.fit(X, Y)
print(model.predict([[1, 10]]))
tree.plot_tree(model, feature_names=['f1', 'f2'], fontsize=10)
plt.show()