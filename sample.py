from sklearn import tree
from matplotlib import pyplot as plt

X = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
y = [0.5, 2.5, 3.5, 4.5, 5.5]
model = tree.DecisionTreeRegressor()
model = model.fit(X, y)
print(model.predict([[1, 10]]))
tree.plot_tree(model)
plt.show()