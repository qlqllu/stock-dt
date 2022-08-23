import os
import pandas as pd
from sklearn import tree
from matplotlib import pyplot as plt
from joblib import dump, load
import time

data_folder = 'E:/github/C3-Data-Science/backtest/datas/stock/zh_a'

stock_list = os.listdir(data_folder)
stock_list = [stock for stock in stock_list if stock.startswith('sh') or stock.startswith('sz')]
test_stocks = stock_list[1000:1100]
# print(train_stocks)
# print(test_stocks)

def build_XY(stocks):
  X, Y = [], []
  for s in stocks:
    stock_file = f'{data_folder}/{s}'

    data = pd.read_csv(stock_file, index_col=0, parse_dates=True)
    data = data.loc[:, ['open', 'close', 'high', 'low', 'volume']]
    data = data.iloc[:7]

    data['price_change_pct'] = round((data.close - data.close.shift(1)) / data.close.shift(1) * 100, 2)
    data.loc[(data['price_change_pct'] > 0), 'up_down'] = 1
    data.loc[(data['price_change_pct'] <= 0), 'up_down'] = 0

    X.append(data['up_down'][1:6].values)
    Y.append(data['up_down'][6])

  return X, Y

test_X, test_Y = build_XY(test_stocks)

model = load('dt2.joblib')
predict_Y = model.predict(test_X)

same_count = 0
for i, val in enumerate(predict_Y):
  if val == test_Y[i]:
    same_count += 1

print(f'result: {round(same_count/len(test_Y)*100, 2)} %')
# print(f'Test Y: {test_Y}')
# print(f'Predict Y: {predict_Y}')

tree.plot_tree(model, fontsize=10)
plt.show()


# print(X)
# print(Y)






