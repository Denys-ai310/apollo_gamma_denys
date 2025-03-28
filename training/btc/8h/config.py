##################################################################################################
### set these parameters before training
##################################################################################################
instrument = 'btc'                       # btc/eth/xrp (the data should be downloaded first)
exchange = 'dydx'                        # bitmex/binance/derabits (atm dydx is supported)
str_time_horizon = '8h'                 # 30min, 1h, 2h, 3h, 4h, 6h, 8h, 12h, 24h 
utc = 0

training_start_date = '2013-01-01'       # must be greator than 2012-01-01
backtest_duration = 150                  # in days
forwardtest_duration = 60                # in days

transaction_fee_percent = 0.05           # transaction fee in percentage 0.0, 0.025, 0.05, ...
take_profit_percent = 100                  # take profit in percentage 1, 2, 3, ... 
stop_loss_percent = 4                    # stop loss in percentage 1, 2, 3, ...
leverage = 1                             # leverage for the instrument 1x, 2x, 3x, ...

use_gpu = True                          # use the gpu for training (if available)

##################################################################################################
### Following indicators can be used as input features for training:
##################################################################################################
###  'vwma', 'sma', 'rsi', 'atr', 'stoch_oscilator', 'bollinger_bands_binary', 
### 'engulfing_candle_binary', 'obv_divergence_binary', 'volume_adi', 'volume_mfi', 
### 'volatility_kcw', 'volatility_kchi', 'volatility_kcli', 'trend_mass_index', 
###  'trend_aroon_up', 'trend_aroon_down', 'trend_psar_up_indicator', 'trend_psar_down_indicator'

###  In case of no indicators, the list_indicators = [] should be used

# ### 24h
# list_indicators = ['vwma', 'rsi', 'bollinger_bands_binary', 'engulfing_candle_binary', 
#                     'obv_divergence_binary', 'volatility_kcli', 'trend_mass_index',  
#                     'trend_aroon_down', 'trend_psar_up_indicator', 'trend_psar_down_indicator' ]
### 8h
list_indicators = ['volatility_bbhi', 'volatility_bbli', 'volatility_kcw',
                    'volatility_kchi', 'volatility_kcli', 'trend_mass_index',       
                    'trend_aroon_up', 'trend_aroon_down', 'trend_psar_up_indicator',
                    'trend_psar_down_indicator']
######################################################################################################

