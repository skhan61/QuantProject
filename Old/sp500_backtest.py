# ### SP500 dependency
# import pandas as pd
# import backtrader as bt
# import backtrader.indicators as btind  # <-- NEW: For adding indicators
# import backtrader.analyzers as btanalyzers
# from pypfopt import EfficientFrontier, expected_returns, risk_models
# import yfinance as yf

# # Data Class for Predictions
# class PandasPredictions(bt.feeds.PandasData):
#     lines = ('signal',)
#     params = (
#         ('signal', -1),
#         ('open', 'FEATURE_open'),
#         ('high', 'FEATURE_high'),
#         ('low', 'FEATURE_low'),
#         ('close', 'FEATURE_close'),
#         ('volume', 'FEATURE_volume')
#     )

# # # Function to optimize weights using PyPortfolioOpt
# def optimize_weights(datas):
#     prices = {}
    
#     for data in datas:
#         s = pd.Series(data.close.array, index=data.datetime.array, name=data._name)
#         prices[data._name] = s

#     df = pd.DataFrame(index=prices[next(iter(prices))].index)

#     for ticker, s in prices.items():
#         df = df.merge(s, left_index=True, right_index=True, \
#             how='left').rename(columns={s.name: ticker})

#     df = df.dropna()

#     mu = expected_returns.mean_historical_return(df)
    
#     S_original = risk_models.sample_cov(df)
    
#     # Regularize the covariance matrix
#     delta = 0.05
#     S = (1 - delta) * S_original + delta * np.eye(S_original.shape[0])
    
#     # Ensure diagonal elements are strictly positive
#     S = S + 1e-6 * np.eye(S_original.shape[0])
    
#     ef = EfficientFrontier(mu, S, solver="SCS", verbose=True)
#     weights = ef.max_sharpe(risk_free_rate=0.005)
#     return ef.clean_weights()


# # Strategy Class
# class TradeAndRebalanceStrategy(bt.Strategy):
#     lines = ('benchmark',)
    
#     params = (
#         ('stop_loss', 0.05),
#         ('take_profit', 0.10),
#         ('benchmark_MA_period', 21)  # <-- NEW: Moving Average period for the benchmark
#     )
    
#     def __init__(self):
#         self.rebalance_days = 0
#         self.max_loss = -0.15
#         self.start_cash = self.broker.get_cash()
#         self.benchmark_data = self.getdatabyname("S&P 500")
#         self.benchmark_MA = btind.SimpleMovingAverage(self.benchmark_data, \
#             period=self.params.benchmark_MA_period)
#         self.orders = {}  # to store buy order references
#         self.atr_dict = {data: btind.ATR(data, period=14) for data in self.datas if data._name != "S&P 500"}


#     def log(self, txt, dt=None):
#         ''' Logging function for the strategy. It logs the date and the message provided. '''
#         dt = dt or self.datas[0].datetime.date(0)
#         print(f"{dt.isoformat()}, {txt}")

#     def notify_order(self, order):
#         # If an order is completed, remove it from the orders dict
#         if order.status == order.Completed:
#             if order.ref in self.orders:
#                 del self.orders[order.ref]

#     def next(self):
#         # Use the moving average of the benchmark for decisions
#         if self.benchmark_data.close[0] > self.benchmark_MA[0] * 1.01:  # Bullish scenario
#             benchmark_trend = 1
#         else:  # Bearish scenario
#             benchmark_trend = -1
            
#         self.log(f"Benchmark Trend: {'Bullish' if benchmark_trend == 1 else 'Bearish'}")

#         benchmark_return = (self.benchmark_data.close[0] - \
#             self.benchmark_data.close[-1]) / self.benchmark_data.close[-1]
#         self.log(f"Benchmark Return: {benchmark_return * 100:.2f}%")

#         if (self.broker.get_cash() - self.start_cash) / self.start_cash <= self.max_loss:
#             return
        
#         for data in self.datas:
#             if data._name == "S&P 500":  # Skip the benchmark for trading signals
#                 continue
#             atr_value = self.atr_dict[data][0] if data in self.atr_dict else 0

#             # Making decisions based on benchmark's performance
#             if benchmark_return > 0:  # Benchmark shows positive returns
#                 if data.signal[0] == 1:
#                     order = self.buy(data)
#                     self.orders[order.ref] = order
                    
#                     # Setting dynamic stop-loss and take-profit levels using ATR
#                     stop_price = data.close[0] - atr_value * 2  # Using 2 times ATR as stop loss
#                     limit_price = data.close[0] + atr_value * 2  # Using 2 times ATR as take profit
                    
#                     self.sell(data=data, exectype=bt.Order.Stop, price=stop_price, parent=order.ref)
#                     self.sell(data=data, exectype=bt.Order.Limit, price=limit_price, parent=order.ref)
                    
#             elif benchmark_return < 0:  # Benchmark shows negative returns
#                 self.sell(data)

#         if self.rebalance_days == 0:
#             weights = optimize_weights([data for data in self.datas if data._name != "S&P 500"])
#             for asset, weight in weights.items():
#                 if weight > 0.30:
#                     weights[asset] = 0.30
            
#             for data in self.datas:
#                 if data._name == "S&P 500":
#                     continue
#                 if data._name in weights:
#                     self.order_target_percent(data, target=weights[data._name])
#                 else:
#                     self.close(data)
#             self.rebalance_days = 20
#         else:
#             self.rebalance_days -= 1


# # Fetch S&P 500 data using yfinance
# def fetch_data(ticker, start_date, end_date):
#     df = yf.download(ticker, start=start_date, end=end_date)
#     return df

# # Assume preds is defined somewhere earlier in your code
# start_date = preds.index.get_level_values(1).min()
# end_date = preds.index.get_level_values(1).max()
# sp500_data = fetch_data('^GSPC', start_date, end_date)

# # Convert it into Backtrader format
# benchmark = bt.feeds.PandasData(dataname=sp500_data, name="S&P 500")

# cerebro = bt.Cerebro()
# cerebro.broker.setcommission(commission=0.001)
# cerebro.addanalyzer(btanalyzers.PyFolio, _name='pyfolio')
# cerebro.adddata(benchmark)

# data_dict = {ticker: preds.xs(ticker) for ticker in preds.index.get_level_values(0).unique()}
# for ticker, data_df in data_dict.items():
#     data = PandasPredictions(dataname=data_df, name=ticker)
#     cerebro.adddata(data)

# cerebro.addstrategy(TradeAndRebalanceStrategy)
# results = cerebro.run()

# # Performance Analysis
# returns, positions, transactions, gross_lev \
#     = results[0].analyzers.pyfolio.get_pf_items()