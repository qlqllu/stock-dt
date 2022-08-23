import os
import pandas as pd
from sklearn import tree
from matplotlib import pyplot as plt
from joblib import dump, load
from utils import build_XY2, get_correct_pert, split_stocks

data_folder = 'E:/github/C3-Data-Science/backtest/datas/stock/zh_a'

stock_list = os.listdir(data_folder)
stock_list = [stock for stock in stock_list if stock.startswith('sh') or stock.startswith('sz')]
train_stocks, test_stocks = split_stocks(stock_list, 50, 100)

train_X, train_Y = build_XY2(data_folder, train_stocks, 200, 7)
test_X, test_Y = build_XY2(data_folder, test_stocks, 200, 7)

model = tree.DecisionTreeClassifier(max_depth=3)
model = model.fit(train_X, train_Y)
predict_Y = model.predict(test_X)

dump(model, 'dt3.joblib')

print(f'result: {get_correct_pert(predict_Y, test_Y)}')
# print('---------------')
# print(train_X)
# print(train_Y)

tree.plot_tree(model, fontsize=10)
plt.show()






