import pandas as pd
import mplfinance as mpf

stock_file = 'E:\\github\\C3-Data-Science\\backtest\\datas\\stock\\zh_a\\sz002005.csv'

daily = pd.read_csv(stock_file, index_col=0, parse_dates=True)
daily = daily.loc[:, ['open', 'close', 'high', 'low', 'volume']]

daily['color'] = daily.apply(lambda x: 'red' if (x.close - x.open) > 0 else 'green', axis=1)
daily['price_change'] = daily.close - daily.close.shift(1)
daily['price_change_pct'] = round((daily.close - daily.close.shift(1)) / daily.close.shift(1) * 100, 2)
print(daily.tail(5))
# plotData = daily.tail(100)
# mpf.plot(plotData, type='candle', marketcolor_overrides=plotData['color'].values)