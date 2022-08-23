import os
import pandas as pd
from sklearn import tree
from matplotlib import pyplot as plt
from joblib import dump, load
from utils import build_XY, get_correct_pert, split_stocks

data_folder = 'E:/github/C3-Data-Science/backtest/datas/stock/zh_a'

stock_list = os.listdir(data_folder)
stock_list = [stock for stock in stock_list if stock.startswith('sh') or stock.startswith('sz')]
train_stocks, test_stocks = split_stocks(stock_list, 100, 500)

train_X, train_Y = build_XY(data_folder, train_stocks, 0, 3)
test_X, test_Y = build_XY(data_folder, test_stocks, 0, 3)

model = tree.DecisionTreeClassifier()
model = model.fit(train_X, train_Y)
predict_Y = model.predict(test_X)

dump(model, 'dt2.joblib')

print(f'result: {get_correct_pert(predict_Y, test_Y)}')

# tree.plot_tree(model, fontsize=10)
# plt.show()






