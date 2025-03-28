import numpy as np
from ta.trend import EMAIndicator, SMAIndicator, ADXIndicator, CCIIndicator, MACD, MassIndex, AroonIndicator, PSARIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel
from ta.volume import AccDistIndexIndicator, MFIIndicator
# from numpy_ext import rolling_apply
import pandas as pd
import requests
import json



def volume_adi(df):
	return AccDistIndexIndicator(
        high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).acc_dist_index()

def volume_mfi(df, window=14):
	return MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=window).money_flow_index()

def volatility_kcw(df, window=14):
	return KeltnerChannel(high=df['high'], low=df['low'], close=df['close'], window=window).keltner_channel_wband()

def volatility_kchi(df, window=14):
	return KeltnerChannel(high=df['high'], low=df['low'], close=df['close'], window=window).keltner_channel_hband_indicator()

def volatility_kcli(df, window=14):
	return KeltnerChannel(high=df['high'], low=df['low'], close=df['close'], window=window).keltner_channel_lband_indicator()

def trend_mass_index(df):
	return MassIndex(high=df['high'], low=df['low'], window_fast=9, window_slow=25).mass_index()

def trend_psar_up_indicator(df):
	return PSARIndicator(high=df['high'], low=df['low'], close=df['close'], step=0.02,
            max_step=0.20,).psar_up_indicator()

def trend_psar_down_indicator(df):
	return PSARIndicator(high=df['high'], low=df['low'], close=df['close'], step=0.02,
            max_step=0.20,).psar_down_indicator()

def trend_aroon_up(df):
	return AroonIndicator(close=df['close'], window=25).aroon_up()

def trend_aroon_down(df):
	# return AroonIndicator(close=df['close'], window=25).aroon_down()
	return AroonIndicator(high=df['high'], low=df['low'], window=25).aroon_down()

def vwma(df_data, window=14):
	return (df_data['close'] * df_data['volume']).rolling(window).sum() / df_data['volume'].rolling(window).sum()

def ema(df, window=14, fillna=False):
	return EMAIndicator(df['close'], window=window, fillna=fillna).ema_indicator()

def sma(df, window=14, fillna=False):
	return SMAIndicator(df['close'], window=window, fillna=fillna).sma_indicator()

def rsi(df_data, window=20):
	return RSIIndicator(close=df_data['close'], window=window).rsi()

def stoch_rsi(df_data, window=20, k=3, d=3):
	return StochRSIIndicator(close=df_data['close'], window=window, smooth1=k, smooth2=d).stochrsi() * 100

def stoch_rsi_d(df_data, window=20, k=3, d=3):
	return StochRSIIndicator(close=df_data['close'], window=window, smooth1=k, smooth2=d).stochrsi_d() * 100

def stoch_rsi_k(df_data, window=20, k=3, d=3):
	return StochRSIIndicator(close=df_data['close'], window=window, smooth1=k, smooth2=d).stochrsi_k() * 100

def adx(df_data, window=14):
	return ADXIndicator(high=df_data["high"], low=df_data["low"], close=df_data["close"], window=window).adx()

def cci(df_data, window=14):
	return CCIIndicator(high=df_data["high"], low=df_data["low"], close=df_data["close"], window=window).cci()

def atr(df_data, window=14):
	return AverageTrueRange(high=df_data["high"], low=df_data["low"], close=df_data["close"], window=window).average_true_range()

def macd(df_data, window_slow=26, window_fast=12, window_sign=9):
	return MACD(close=df_data["close"], window_slow=window_slow, window_fast=window_fast, window_sign=window_sign).macd_signal()

def stoch_oscilator(df_data, window=14, smooth_window=3):
	return StochasticOscillator(high=df_data["high"], low=df_data["low"], close=df_data["close"], window=window, smooth_window=smooth_window).stoch_signal()

def bollinger_bands(df_data, window=20, stdev=2):
	bb = BollingerBands(close=df_data["close"], window=window, window_dev=stdev)
	lband_indicator = bb.bollinger_lband_indicator()
	hband_indicator = bb.bollinger_hband_indicator()
	
	return np.where((hband_indicator == 1), 1, (np.where(lband_indicator == 1, -1, 0)))

def engulfing_candle(df_data):
	
	bullish_engulfing = np.logical_and.reduce([
		df_data["close"].shift() < df_data["open"].shift(),  # Previous candle is bearish
		df_data["close"] > df_data["open"],                  # Current candle is bullish
		df_data["close"] > df_data["open"].shift(),          # Current candle engulfs previous candle
		df_data["open"] <= df_data["close"].shift()          # Current candle engulfs previous candle
	])

	bearish_engulfing = np.logical_and.reduce([
		df_data["close"].shift() > df_data["open"].shift(),  # Previous candle is bullish
		df_data["close"] < df_data["open"],                  # Current candle is bearish
		df_data["close"] < df_data["open"].shift(),          # Current candle engulfs previous candle
		df_data["open"] >= df_data["close"].shift()          # Current candle engulfs previous candle
	])

	# Initialize the is_engulfing column with 0s
	df_data["engulfing_candle"] = 0

	# Set 1 for engulfing candles
	df_data.loc[bullish_engulfing, "engulfing_candle"] = 1
	df_data.loc[bearish_engulfing, "engulfing_candle"] = -1

	return df_data["engulfing_candle"]

def rsi_divergence(df_data, rsi_window=14, rsi_low=30, rsi_high=70, width=5):
	df_data["rsid"] = rsi(df_data, rsi_window)
	df_data["rsi_divergence"] = 0

	# Bullish Divergence
	for i in range(rsi_window, len(df_data)):
		try:
			if df_data.iloc[i]["rsid"] < rsi_low:
				for a in range(i + 1, i + width):
					if df_data.iloc[a]["rsid"] > rsi_low:
						for r in range(a + 1, a + width):
							if df_data.iloc[r]["rsid"] < rsi_low and \
								df_data.iloc[r]["rsid"] > df_data.iloc[i]["rsid"] and df_data.iloc[r]["close"] < df_data.iloc[i]["close"]:
									for s in range(r + 1, r + width): 
										if df_data.iloc[s]["rsid"] > rsi_low:
											df_data.iloc[s+1]["rsi_divergence"] = 1
											break
										else:
											continue
							else:
								continue
					else:
						continue
			else:
				continue

		except:
			pass

	# Bearish Divergence
	for i in range(len(df_data)):
		try:
			if df_data.iloc[i]["rsid"] > rsi_high:
				for a in range(i + 1, i + width): 
					if df_data.iloc[a]["rsid"] < rsi_high:
						for r in range(a + 1, a + width):
							if df_data.iloc[r]["rsid"] > rsi_high and \
							df_data.iloc[r]["rsid"] < df_data.iloc[i]["rsid"] and df_data.iloc[r]["close"] > df_data.iloc[i]["close"]:
								for s in range(r + 1, r + width):
									if df_data.iloc[s]["rsid"] < rsi_high:
										df_data.iloc[s+1]["rsi_divergence"] = -1
										break
									else:
										continue
							else:
								continue
					else:
						continue
			else:
				continue
		except:
			pass

	return df_data["rsi_divergence"]

def obv_divergence(df_data):

	# Optimized Calculations
	obv = np.zeros(len(df_data))
	close_prices = df_data["close"].to_numpy()
	volumes = df_data["volume"].to_numpy()

	price_diff = np.sign(close_prices[1:] - close_prices[:-1])
	volume_diff = np.where(price_diff > 0, volumes[1:], np.where(price_diff < 0, -volumes[1:], 0))

	obv[1:] = np.cumsum(volume_diff)
	df_data["obv"] = obv
	df_data["obv_ema"] = df_data["obv"].ewm(span=20).mean()
	df_data["obv_divergence"] = np.where((df_data["obv"] > df_data["obv_ema"]), 1, (np.where((df_data["obv"] < df_data["obv_ema"]), -1, 0)))

	return df_data["obv_divergence"]

