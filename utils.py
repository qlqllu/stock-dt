import pandas as pd

# use rise and fall
def build_XY(data_folder, stocks, start_index, days):
  X, Y = [], []
  for s in stocks:
    stock_file = f'{data_folder}/{s}'

    data = pd.read_csv(stock_file, index_col=0, parse_dates=True)
    data = data.loc[:, ['open', 'close', 'high', 'low', 'volume']]
    data = data.iloc[start_index : (start_index + days + 2)]

    data['price_change'] = data.close - data.close.shift(1)
    data.loc[(data['price_change'] > 0), 'up_down'] = 1
    data.loc[(data['price_change'] <= 0), 'up_down'] = 0

    X.append(data['up_down'][1 : (days + 1)].values)
    Y.append(data['up_down'][days + 1])

  return X, Y

# use count of rise
def build_XY2(data_folder, stocks, start_index, days):
  X, Y = [], []
  for s in stocks:
    stock_file = f'{data_folder}/{s}'

    data = pd.read_csv(stock_file, index_col=0, parse_dates=True)
    data = data.loc[:, ['open', 'close', 'high', 'low', 'volume']]
    data = data.iloc[start_index : (start_index + days + 2)]

    if data.shape[0] == 0:
      continue

    data['price_change'] = data.close - data.close.shift(1)
    data.loc[(data['price_change'] > 0), 'up_down'] = 1
    data.loc[(data['price_change'] <= 0), 'up_down'] = 0

    x_values = data['up_down'][1 : (days + 1)].values
    X.append([len(list(filter(lambda x: x == 1, x_values)))])
    Y.append(data['up_down'][days + 1])

  return X, Y

# use open and close
def build_XY3(data_folder, stocks, start_index, days):
  X, Y = [], []
  for s in stocks:
    stock_file = f'{data_folder}/{s}'

    data = pd.read_csv(stock_file, index_col=0, parse_dates=True)
    data = data.loc[:, ['open', 'close', 'high', 'low', 'volume']]
    data = data.iloc[start_index : (start_index + days + 2)]

    data['price_change'] = data.close - data.close.shift(1)
    data['oc_change'] = data.close - data.open
    data.loc[(data['oc_change'] > 0), 'yy'] = 1
    data.loc[(data['oc_change'] <= 0), 'yy'] = 0
    data.loc[(data['price_change'] > 0), 'up_down'] = 1
    data.loc[(data['price_change'] <= 0), 'up_down'] = 0

    X.append(data['yy'][1 : (days + 1)].values)
    Y.append(data['up_down'][days + 1])

  return X, Y

def get_correct_pert(predict_Y, test_Y):
  same_count = 0
  for i, val in enumerate(predict_Y):
    if val == test_Y[i]:
      same_count += 1

  return round(same_count/len(test_Y)*100, 2)

def split_stocks(stock_list, train_count, test_count):
  return stock_list[0 : train_count], stock_list[train_count + 1 : train_count + 1 + test_count]