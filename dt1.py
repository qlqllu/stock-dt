from genericpath import isdir
import os

data_folder = 'E:\\github\\C3-Data-Science\\backtest\\datas\\stock\\zh_a'
start_date = '2022-01-01'

files = os.listdir(data_folder)
train_stocks, test_stocks = files[:10], files[11:12]
print(train_stocks)
print(test_stocks)



