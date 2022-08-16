import pandas as pd
import mplfinance as mpf

stock_file = 'E:\\github\\C3-Data-Science\\backtest\\datas\\stock\\zh_a\\sh600571.csv'

daily = pd.read_csv(stock_file, index_col=0, parse_dates=True)
daily['color'] = daily.apply(lambda x: 'red' if (x.close - x.open) > 0 else 'green', axis=1)
# print(daily.head(3)['color'].values)
plotData = daily.tail(100)
mpf.plot(plotData, type='candle', marketcolor_overrides=plotData['color'].values)