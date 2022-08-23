import pandas as pd

stock_file = 'E:/github/C3-Data-Science/backtest/datas/stock/zh_a/sz002005.csv'

data = pd.read_csv(stock_file, index_col=0, parse_dates=True)

# print(data.open)
# print(data.open.values)
# print(data.loc[:, 'open'])
# print(data.loc[:, ['open', 'close']])
# print(data.loc['2004-06-28', ['open', 'close']])
# print(data.loc['2004-06-28':'2004-06-30', ['open', 'close']])
# print(data.loc['2004-06-28':'2004-06-30'])

# print(data.iloc[:, 0])
# print(data.iloc[:, [0, 1]])
# print(data.iloc[:1, [0, 1]])
# print(data.iloc[1:3, [0, 1]])
# print(data.iloc[0:5])