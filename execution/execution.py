# import config
import indicators_utils as custom_indicators
from stable_baselines3 import PPO, TD3, DDPG,  SAC, A2C
import os
import sys
# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from learners import learner_drl
from utils import paths, Dataset, utils
from drl_envs.env_ld_sw_dataset import EnvTrading
import os
import torch
import sys
import numpy as np
import random
import pandas as pd
# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from app.backtest import backtest
import quantstats as qs
import glob
# set random seeds in random module, numpy module and PyTorch module.
seed = 818
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# torch.manual_seed(torch.initial_seed())


class Execution:
	def __init__(self, data, model_path):
		base_name = os.path.basename(model_path)
		self.lib = base_name.split('_')[2]
		algo = base_name.split('_')[3]
		self.rolling_window_size = int(base_name.split('_')[4].split('.')[0])
		
		# if lib == 'd3rlpy':
		# 	model_name_prefix = base_name.replace('.pt', '')
		# else:
		# 	model_name_prefix = base_name.replace('.zip', '')

		self.model = self.load_custom_model(model_path, self.lib, algo)
		# print(self.model)
		# print(list(self.model.parameters())[0].shape)

		# #################################################################################################
		# ### set these parameters before training
		# #################################################################################################
		# instrument = 'btc'                       # btc/eth/xrp (the data should be downloaded first)
		# exchange = 'dydx'                        # bitmex/binance/derabits (atm dydx is supported)
		# str_time_horizon = '24h'                 # 30min, 1h, 2h, 3h, 4h, 6h, 8h, 12h, 24h 
		# utc = 0

		# training_start_date = '2023-01-01'       # must be greator than 2012-01-01
		# backtest_duration = 60                  # in days
		# forwardtest_duration = 30                # in days

		# self.rolling_window_size = 9

		self.list_indicators = ['vwma', 'rsi', 'bollinger_bands_binary', 'engulfing_candle_binary', 
							'obv_divergence_binary', 'volatility_kcli', 'trend_mass_index',  
							'trend_aroon_down', 'trend_psar_up_indicator', 'trend_psar_down_indicator' ]
								
		


		# current_dir = os.path.dirname(os.path.realpath(__file__))

		# self.data = pd.read_csv('E:/apollo_gamma/data/dydx/btc/btc_24h_2012_onwards_utc0.csv')
		self.data = data
		self.data['datetime'] = pd.to_datetime(self.data['datetime'], utc=True)
		self.data.set_index('datetime', inplace=True)
		self.data.index = pd.to_datetime(self.data.index, utc=True)
		# print(self.data)



		if len(self.list_indicators) > 0:
			self.indicators = True
			self.binary_indicators = False
			self.add_custom_indictors()

		# print(self.data)

		self.input_data = self.create_input()
		# print(self.input_data)

		######################################################################################################


	def create_input(self):
		# print('\n', df_type)
		df = self.data.copy()
		close_only = self.data[['close']].copy()

		##############################################################################
		### create a new dataframe based on rolling window size
		df_rolling_window = self.np_create_rolling_window(df, self.rolling_window_size)
		# print(df_rolling_window.columns)
		# print(df_rolling_window)

		### take log difference of df_rolling_window except for columns containing binary or negative values
		# # Identify columns with binary values or negative values
		# binary_cols = df_rolling_window.columns[df_rolling_window.isin([0, 1]).all()]
		# negative_cols = df_rolling_window.columns[(df_rolling_window <= 0).any()]

		df_train_log_transformed = df_rolling_window.copy()

		list_non_logged_cols = []
		list_negative_cols = []
		# Calculate log difference only for non-binary and non-negative columns
		for column in df_rolling_window.columns:
			# if column not in binary_cols and column not in negative_cols:
				# df_train_log_transformed[column] = np.log(df_rolling_window[column]).diff()
			if all(df_rolling_window[column].isin([-1, 0, 1])) or\
				all(df_rolling_window[column].isin([-1, 1])) or\
				all(df_rolling_window[column].isin([1, 0])):
				list_non_logged_cols.append(column)

			elif any(df_rolling_window[column] == 0) or any(df_rolling_window[column] < 0):
				list_negative_cols.append(column)
			
			else:
				df_train_log_transformed[column] = np.log(df_rolling_window[column]).diff()


		# print(list_negative_cols)
		if len(list_negative_cols) > 0:
			df_train_log_transformed = df_train_log_transformed.drop(columns=list_negative_cols)
		# print(df_train_log_transformed)

		### find if there are any missing column numbers, an fix it.
		max_column_index = max(map(int, df_train_log_transformed.columns), default=-1)
		missing_columns = [str(i) for i in range(max_column_index + 1) if str(i) not in df_train_log_transformed.columns]
		mapping = {col: str(i) for i, col in enumerate(df_train_log_transformed.columns)}
		df_train_log_transformed.rename(columns=mapping, inplace=True)



		# print(df_train_log_transformed)
		df_train_log_transformed.dropna(inplace=True)
		# print(df_train_log_transformed)

		# ### create target, i.e., close only values
		# df_target_log_transformed = np.log(close_only) - np.log(close_only.shift(1)) 
		# df_target_log_transformed = df_target_log_transformed.shift(-1)
		# df_target_log_transformed = df_target_log_transformed.reindex(df_train_log_transformed.index)
		# df_target_log_transformed.dropna(inplace=True)
		# df_target_log_transformed.rename(columns={'close': 'target'}, inplace=True)
		# df_target_log_transformed['direction'] = np.sign(df_target_log_transformed['target'])
		# # print(df_target_log_transformed)

		# ### reindex df_log_transformed to match the length of df_close_log_transformed (target)
		# df_train_log_transformed = df_train_log_transformed.reindex(df_target_log_transformed.index)
		# # print(df_train_log_transformed)

		

		# df_bt_ = df_target_log_transformed.copy()
		# df_bt_.index = pd.DatetimeIndex(df_bt_.index) + pd.DateOffset(minutes=self.time_horizon_minutes)    
		# # df_bt_['target'] = np.sign(df_bt_['target'])
		# # df_bt_.rename(columns={'target':'actual_direction'}, inplace=True)
		# # # print(df_bt_)
		

		# # print(df_bt)
		# if df_type == 'backtest':
		# 	self.df_backtest = df_target_log_transformed.copy()
		# 	self.df_backtest.index = pd.DatetimeIndex(self.df_backtest.index) + pd.DateOffset(minutes=self.time_horizon_minutes)   
		# 	# print(self.df_backtest)

		# elif df_type == 'forwardtest':
		# 	self.df_forwardtest = df_target_log_transformed.copy()
		# 	self.df_forwardtest.index = pd.DatetimeIndex(self.df_forwardtest.index) + pd.DateOffset(minutes=self.time_horizon_minutes)   
		# 	# print(self.df_forwardtest)
					
		# # Check for NaN values
		# has_nan_or_inf = df_train_log_transformed.isna().any().any() or np.isinf(df_train_log_transformed.values).any()
		# if has_nan_or_inf:
		# 	print("There are NaN/inf values in the DataFrame.")
		# 	exit(0)

		# # print(df_train_log_transformed)
		# # print(df_target_log_transformed)
		# return df_train_log_transformed, df_target_log_transformed
		return df_train_log_transformed.iloc[-1]

	def np_create_rolling_window(self, df, rw_size):
	
		# Create a sliding window view of the DataFrame values
		window_view = np.lib.stride_tricks.sliding_window_view(df.values, (rw_size, df.shape[1]))
		# Reshape the sliding window view to a 2D array
		new_arr = window_view.reshape(-1, df.shape[1] * rw_size)
		# Create a new DataFrame with the sliding window values
		new_df = pd.DataFrame(new_arr)
		new_df.columns = range(df.shape[1] * rw_size)

		# print(new_df)
		# Assign numeric indices to columns that do not contain the keyword 'binary'
		# new_df.columns = [str(i) if 'binary' not in col else col for i, col in enumerate(new_df.columns)]

		# Copy the DateTime index from the original DataFrame to the new DataFrame
		new_df.index = df.index[rw_size - 1:]
		
		return new_df

	def add_custom_indictors(self):		
		df_data = self.data.copy()
		self.list_binary_indicators = []

		if 'vwma' in self.list_indicators:
			df_data['vwma'] = custom_indicators.vwma(df_data.copy())

		if 'sma' in self.list_indicators:
			df_data['sma'] = custom_indicators.sma(df_data.copy())

		if 'rsi' in self.list_indicators:
			df_data['rsi'] = custom_indicators.rsi(df_data.copy())

		if 'atr' in self.list_indicators:
			df_data['atr'] = custom_indicators.atr(df_data.copy())

		if 'stoch_oscilator' in self.list_indicators:
			df_data['stoch_oscilator'] = custom_indicators.stoch_oscilator(df_data.copy())

		if 'bollinger_bands_binary' in self.list_indicators:
			df_data['bollinger_bands_binary'] = custom_indicators.bollinger_bands(df_data.copy())

		if 'engulfing_candle_binary' in self.list_indicators:
			df_data['engulfing_candle_binary'] = custom_indicators.engulfing_candle(df_data.copy())

		if 'obv_divergence_binary' in self.list_indicators:
			df_data['obv_divergence_binary'] = custom_indicators.obv_divergence(df_data.copy())

		if 'volume_adi' in self.list_indicators:
			df_data['volume_adi'] = custom_indicators.volume_adi(df_data.copy())

		if 'volume_mfi' in self.list_indicators:
			df_data['volume_mfi'] = custom_indicators.volume_mfi(df_data.copy())

		if 'volatility_kcw' in self.list_indicators:
			df_data['volatility_kcw'] = custom_indicators.volatility_kcw(df_data.copy())

		if 'volatility_kchi' in self.list_indicators:
			df_data['volatility_kchi'] = custom_indicators.volatility_kchi(df_data.copy())

		if 'volatility_kcli' in self.list_indicators:
			df_data['volatility_kcli'] = custom_indicators.volatility_kcli(df_data.copy())

		if 'trend_mass_index' in self.list_indicators:
			df_data['trend_mass_index'] = custom_indicators.trend_mass_index(df_data.copy())

		if 'trend_aroon_up' in self.list_indicators:
			df_data['trend_aroon_up'] = custom_indicators.trend_aroon_up(df_data.copy())

		if 'trend_aroon_down' in self.list_indicators:
			df_data['trend_aroon_down'] = custom_indicators.trend_aroon_down(df_data.copy())

		if 'trend_psar_up_indicator' in self.list_indicators:
			df_data['trend_psar_up_indicator'] = custom_indicators.trend_psar_up_indicator(df_data.copy())

		if 'trend_psar_down_indicator' in self.list_indicators:
			df_data['trend_psar_down_indicator'] = custom_indicators.trend_psar_down_indicator(df_data.copy())

				
		df_data.dropna(inplace=True)
		self.data = df_data

	def get_prediction(self):
		input_data = [self.input_data]
		if self.lib == 'sb3':
			# print(input_data, tmp_model)
			input_data = np.array(input_data, dtype= np.float32)
			model_predictions = np.sign(np.ravel(self.model.predict(input_data, deterministic=True)[0]))
		elif self.lib == 'd3rlpy':
			obs = np.array(input_data, dtype= np.float32)
			obs = torch.from_numpy(obs)
			model_predictions = np.sign(np.ravel(self.model(obs)))


		# df_backtest['predicted_direction'] = np.sign(model_predictions)
		# df_pred = df_backtest.dropna()
		# predictions = df_pred

		# obj_backtest = backtest.Backtest(predictions, instrument_exchange_timehorizon_utc, transaction_fee=0.0)
		# ledger, ending_balance, sharpe, r2, pnl_percent = obj_backtest.run()
		# r2 = round(r2, 2)

		# # print(ledger)
		# print(ending_balance, sharpe, r2, pnl_percent)

		

		return model_predictions[0]

	def load_custom_model(self, model_path, lib, algo):
		if lib == 'sb3':
			from stable_baselines3 import PPO, TD3, DDPG,  SAC, A2C
			if algo == 'ddpg':
				model = DDPG.load(model_path)
			elif algo == 'ppo':
				model = PPO.load(model_path)
			elif algo == 'td3':
				model = TD3.load(model_path)
			elif algo == 'sac':
				model = SAC.load(model_path)
			elif algo == 'a2c':
				model = A2C.load(model_path)

		elif lib == 'd3rlpy':
			model = torch.jit.load(model_path)

		return model








data = pd.read_csv('E:/apollo_gamma/data/dydx/btc/btc_24h_2012_onwards_utc0.csv')
# data['datetime'] = pd.to_datetime(self.data['datetime'], utc=True)


# model_path = 'E:/apollo_gamma/training/btc/24h/models/BTC_24h_d3rlpy_bear_29.pt'
# base_name = os.path.basename(model_path)

# lib = base_name.split('_')[-3]
# rolling_window_size = int(base_name.split('_')[-1].split('.')[0])
# print(base_name)

# obj_dataset = Execution(data.copy(), model_path)
# print(obj_dataset.get_prediction())
# del obj_dataset



current_dir = os.path.dirname(os.path.realpath(__file__))
models_directory = os.path.join(current_dir, 'models')

list_of_models = glob.glob(models_directory + '/*')
# print(list_of_models)

for model_path in list_of_models:

	# model_path = 'E:/apollo_gamma/training/btc/24h/models/BTC_24h_d3rlpy_bear_29.pt'
	# base_name = 'BTC_24h_d3rlpy_bear_29.pt'
	base_name = os.path.basename(model_path)

	lib = base_name.split('_')[-3]
	# algo = base_name.split('_')[-2]
	rolling_window_size = int(base_name.split('_')[4].split('.')[0])

	if '.html' in base_name:
		continue
	
	# if lib == 'd3rlpy':
	# 	continue

	# if rolling_window_size == 29 or rolling_window_size == 1:
	# 	continue
	print(base_name)

	obj_dataset = Execution(data.copy(), model_path)
	print(obj_dataset.get_prediction())
	del obj_dataset
