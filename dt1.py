import os
import pandas as pd
from sklearn import tree
from matplotlib import pyplot as plt

data_folder = 'E:/github/C3-Data-Science/backtest/datas/stock/zh_a'

files = os.listdir(data_folder)
files = [file for file in files if file.startswith('sh') or file.startswith('sz')]
train_stocks, test_stocks = files[0:100], files[1200:1350]
# print(train_stocks)
# print(test_stocks)

def build_XY(stocks):
  X, Y = [], []
  for s in stocks:
    stock_file = f'{data_folder}/{s}'
    data = pd.read_csv(stock_file, index_col=0, parse_dates=True)
    data = data.loc[:, ['open', 'close', 'high', 'low', 'volume']]
    data = data.iloc[:12]

    data['price_change_pct'] = round((data.close - data.close.shift(1)) / data.close.shift(1) * 100, 2)
    data.loc[(data['price_change_pct'] > 0), 'up_down'] = 1
    data.loc[(data['price_change_pct'] <= 0), 'up_down'] = 0

    X.append(data['up_down'][1:11].values)
    Y.append(data['up_down'][11])

  return X, Y

train_X, train_Y = build_XY(train_stocks)
test_X, test_Y = build_XY(test_stocks)

model = tree.DecisionTreeClassifier()
model = model.fit(train_X, train_Y)
predict_Y = model.predict(test_X)

same_count = 0
for i, val in enumerate(predict_Y):
  if val == test_Y[i]:
    same_count += 1

print(f'result: {round(same_count/len(test_Y)*100, 2)}')
print(f'Test Y: {test_Y}')
print(f'Predict Y: {predict_Y}')

# tree.plot_tree(model, fontsize=10)
# plt.show()


# print(X)
# print(Y)






