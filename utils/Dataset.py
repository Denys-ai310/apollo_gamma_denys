import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.utils import dropna
import os
import sys
# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils import utils as ut
from utils import indicators_utils as custom_indicators
import os

import warnings
warnings.filterwarnings('ignore')



class Dataset:
	def __init__(self, instrument, time_horizon, data_path, split_triple, list_indicators, rolling_window_size=1, retrain=False, trained_until='20Sep22'):
		self.instrument = instrument
		self.str_time_horizon = time_horizon
		self.time_horizon_minutes = (int(self.str_time_horizon.split('h')[0]) * 60 ) if 'h' in self.str_time_horizon else int(self.str_time_horizon.split('min')[0]) 
		self.rolling_window_size = rolling_window_size
		self.list_indicators = list_indicators
		self.correlation_threshold = 0.65

		if os.path.exists(data_path):
			self.data = pd.read_csv(data_path)
			# Rename columns to match expected format
			self.data = self.data.rename(columns={
				'Date': 'datetime',
				'Low': 'low',
				'High': 'high',
				'Open': 'open',
				'Close': 'close',
				'Volume': 'volume'
			})
			self.data['datetime'] = pd.to_datetime(self.data['datetime'], utc=True)
			start_date = pd.to_datetime(split_triple[0], utc=True) - pd.Timedelta(days=30)
			self.data = self.data.loc[(self.data.datetime >= start_date)]			
		else:
			print(f'Data not found, please create data for {self.str_time_horizon}')
			exit(0)

		#### np conversion 
		self.np_data = self.data.to_numpy()
		self.np_data_date_index = self.data.columns.get_loc("datetime")
		self.np_data_open_index = self.data.columns.get_loc("open")
		self.np_data_high_index = self.data.columns.get_loc("high")
		self.np_data_low_index = self.data.columns.get_loc("low")
		self.np_data_close_index = self.data.columns.get_loc("close")
		self.np_data_volume_index = self.data.columns.get_loc("volume")

		##################################################

		self.data.set_index('datetime', inplace=True)
		self.data.index = pd.to_datetime(self.data.index, utc=True)

		input_features_basic = ['open', 'high', 'low', 'close', 'volume']
		self.data = self.data[input_features_basic]


		if len(self.list_indicators) > 0:
			self.indicators = True
			self.binary_indicators = False
			self.add_custom_indictors()

		if retrain:
			train, backtest, forwardtest = ut.split_train_test_rw_retrain(self.data.copy(), split_triple, self.time_horizon_minutes*self.rolling_window_size, trained_until)
		else:
			train, backtest, forwardtest = ut.split_train_test_rw(self.data.copy(), split_triple, self.time_horizon_minutes*self.rolling_window_size)
		# print('*'*100, 'train, backtest, forwardtest')
		# print('\ntrain', train)
		# print('\nbacktest', backtest)
		# print('\nforwardtest', forwardtest)
		# print('*'*100)

		self.X_train, self.y_train = self.create_dataset(train, 'train')
		self.X_backtest, self.y_backtest = self.create_dataset(backtest, 'backtest')
		self.X_forwardtest, self.y_forwardtest = self.create_dataset(forwardtest, 'forwardtest')

		self.train_test_details = str((self.y_train.index[-1]).date().strftime("%d%b%y")) + '_' + str(len(self.X_backtest)-self.rolling_window_size) + '_' + str(len(self.X_forwardtest)-self.rolling_window_size)
		# print()

	def create_dataset(self, df_tmp, df_type):
		# print('\n', df_type)
		df = df_tmp.copy()
		close_only = df_tmp[['close']].copy()

		##############################################################################
		### create a new dataframe based on rolling window size
		df_rolling_window = ut.np_create_rolling_window(df, self.rolling_window_size)
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

		### create target, i.e., close only values
		df_target_log_transformed = np.log(close_only) - np.log(close_only.shift(1)) 
		df_target_log_transformed = df_target_log_transformed.shift(-1)
		df_target_log_transformed = df_target_log_transformed.reindex(df_train_log_transformed.index)
		df_target_log_transformed.dropna(inplace=True)
		df_target_log_transformed.rename(columns={'close': 'target'}, inplace=True)
		df_target_log_transformed['direction'] = np.sign(df_target_log_transformed['target'])
		# print(df_target_log_transformed)

		### reindex df_log_transformed to match the length of df_close_log_transformed (target)
		df_train_log_transformed = df_train_log_transformed.reindex(df_target_log_transformed.index)
		# print(df_train_log_transformed)

		

		df_bt_ = df_target_log_transformed.copy()
		df_bt_.index = pd.DatetimeIndex(df_bt_.index) + pd.DateOffset(minutes=self.time_horizon_minutes)    
		# df_bt_['target'] = np.sign(df_bt_['target'])
		# df_bt_.rename(columns={'target':'actual_direction'}, inplace=True)
		# # print(df_bt_)
		

		# print(df_bt)
		if df_type == 'backtest':
			self.df_backtest = df_target_log_transformed.copy()
			self.df_backtest.index = pd.DatetimeIndex(self.df_backtest.index) + pd.DateOffset(minutes=self.time_horizon_minutes)   
			# print(self.df_backtest)

		elif df_type == 'forwardtest':
			self.df_forwardtest = df_target_log_transformed.copy()
			self.df_forwardtest.index = pd.DatetimeIndex(self.df_forwardtest.index) + pd.DateOffset(minutes=self.time_horizon_minutes)   
			# print(self.df_forwardtest)
					
		# Check for NaN values
		has_nan_or_inf = df_train_log_transformed.isna().any().any() or np.isinf(df_train_log_transformed.values).any()
		if has_nan_or_inf:
			print("There are NaN/inf values in the DataFrame.")
			exit(0)

		# print(df_train_log_transformed)
		# print(df_target_log_transformed)

		# Before returning, if this is for DRL training/prediction, 
		# we need to ensure the data is in the correct shape
		
		
		# if hasattr(self, 'learner') and self.learner == 'drl':
		# 	# Select only the first 13 columns if we have more
		# 	if df_train_log_transformed.shape[1] > 13:
		# 		df_train_log_transformed = df_train_log_transformed.iloc[:, :13]
		# 	# Pad with zeros if we have fewer than 13 columns
		# 	elif df_train_log_transformed.shape[1] < 13:
		# 		missing_cols = 13 - df_train_log_transformed.shape[1]
		# 		for i in range(missing_cols):
		# 			df_train_log_transformed[f'padding_{i}'] = 0
				
		# 	# Take only the most recent observation
		# 	df_train_log_transformed = df_train_log_transformed.iloc[-1:, :]
			
		# 	# Convert to numpy array and reshape to (13,) for single prediction
		# 	df_train_log_transformed = df_train_log_transformed.to_numpy().flatten()

		return df_train_log_transformed, df_target_log_transformed

		
		
	def create_train_target(self, tmp_df, input_str):
		### df with rolling window data
		df = tmp_df.copy()
		close_only = tmp_df[['close']].copy()
		df_ohlv = tmp_df[[col for col in tmp_df.columns if col not in ['close']]]
		# print(df)
			
		##############################################################################
		### create a new dataframe based on rolling window size
		df_rolling_window = ut.np_create_rolling_window(df, self.rolling_window_size)
		print(df_rolling_window)

		### take log difference of df_rolling_window except for columns containing binary or negative values
		# Identify columns with binary values or negative values
		binary_cols = df_rolling_window.columns[df_rolling_window.isin([0, 1]).all()]
		int_cols = df_rolling_window.columns[df_rolling_window.isin([-1, 0, 1]).all()]
		negative_cols = df_rolling_window.columns[(df_rolling_window <= 0).any()]

		df_train_log_transformed = df_rolling_window.copy()

		# Calculate log difference only for non-binary and non-negative columns
		for column in df_rolling_window.columns:
			if column not in binary_cols and column not in negative_cols:
				df_train_log_transformed[column] = np.log(df_rolling_window[column]).diff()

		df_train_log_transformed.dropna(inplace=True)

		### create target, i.e., close only values
		df_target_log_transformed = np.log(close_only) - np.log(close_only.shift(1)) 
		df_target_log_transformed = df_target_log_transformed.shift(-1)
		df_target_log_transformed = df_target_log_transformed.reindex(df_train_log_transformed.index)
		df_target_log_transformed.dropna(inplace=True)
		df_target_log_transformed.rename(columns={'close': 'target'}, inplace=True)
		df_target_log_transformed['direction'] = np.sign(df_target_log_transformed['target'])
		# print(df_target_log_transformed)

		### reindex df_log_transformed to match the length of df_close_log_transformed (target)
		df_train_log_transformed = df_train_log_transformed.reindex(df_target_log_transformed.index)
		# print(df_train_log_transformed)

		# print(self.df_data)
		
		# print(self.df_data)


		# print(df_train_log_transformed)
		# print(df_target_log_transformed)



		

		
		# df_without_binary = df.drop('obv_divergence_binary', axis=1)
		# # print(df_without_binary)
		# df = ut.create_rolling_window(df_without_binary, self.rolling_window_size)
		# # print(df)
		# df_log_differenced = ut.df_log_difference(df.copy())
		# print(df_log_differenced)

		# # close_only_log_differenced = np.log(close_only) - np.log(close_only.shift(1)) 
		# # close_only_log_differenced = close_only_log_differenced.shift(-1)
		# # close_only_log_differenced = close_only_log_differenced.reindex(log_diff_df.index)
		# # close_only_log_differenced.dropna(inplace=True)

		# # log_diff_df = log_diff_df.reindex(close_only_log_differenced.index)
		# # close_only_log_differenced.index = pd.DatetimeIndex(close_only_log_differenced.index) + pd.DateOffset(hours=self.time_horizon)    
		# # print(close_only_log_differenced)
		# # print(log_diff_df)




		# ### create log differenced train/target/close_only dataframes

		# train = df_log_differenced[[col for col in df_log_differenced.columns if col not in ['actual_close', 'next_close', 'actual_close_log_diff', 'next_close_log_diff',]]]
		# target = pd.DataFrame(df_log_differenced[['next_close_log_diff']]).rename(columns={'next_close_log_diff':'target'})
		# train_darts = ut.df_log_difference(close_only) ### since it will be used for darts training, it does not need to be shifted (+1)
		# df_ohlv = ut.df_log_difference(df_ohlv) ### since it will be used for darts training, it does not need to be shifted (+1)

		# # print('*'*100, input_str)
		# # # print(tmp_df)
		# # print(train)
		# # print(target)

		# # if self.indicators:
		# # 	if self.binary_indicators:
		# # 		start_date = train.index[0]
		# # 		end_date = train.index[-1]
		# # 		df_data_dropped_tmp = self.df_data_dropped[(self.df_data_dropped.index >= start_date) & (self.df_data_dropped.index <= (end_date))]
		# # 		# print(self.df_data_dropped)

		# # 		last_column_name = int(train.columns[-1])
		# # 		num_columns = df_data_dropped_tmp.shape[1]
		# # 		for i in range(num_columns):
		# # 			train[str(last_column_name+i+1)] = df_data_dropped_tmp.iloc[:, i]

		# # 		train.columns = train.columns.astype(str)

		# 	# print(train)

		# # print(train_darts) 
		# # print(df_ohlv) 
		# # if input_str == 'train':
		# #     print(train_darts)

		# # self.learner = 'timeseries_darts'

		# self.learner = 'classification'
		# ### modify train/target df according to the learner type, such as classification etc. 
		# if self.learner == 'classification':
		# 	target = np.sign(target)
		# 	target.loc[target.target == 0, "target"] = 1

		# elif self.learner == 'drl':
		# 	target.loc[:, 'direction'] = np.sign(target[['target']])
		# 	target.loc[target.direction == 0, "direction"] = 1

		# elif self.learner == 'timeseries_darts':
		# 	from darts import TimeSeries, concatenate
		# 	train_darts = train_darts.rename_axis('datetime').reset_index()
		# 	train_darts['datetime'] = pd.to_datetime(train_darts.datetime).dt.tz_localize(None)
		# 	train = TimeSeries.from_dataframe(train_darts, time_col='datetime', value_cols='close')

		# 	df_ohlv = df_ohlv.rename_axis('datetime').reset_index()
		# 	df_ohlv['datetime'] = pd.to_datetime(df_ohlv.datetime).dt.tz_localize(None)
		# 	# if input_str == 'train':
		# 	#     self.df_ohlv = TimeSeries.from_dataframe(df_ohlv, time_col='datetime', value_cols='open')
		# 	#     print(df_ohlv)

		# ### create backtest/realworld dataframe
		# df_bt = pd.DataFrame(df_log_differenced[['actual_close', 'next_close']]).rename(columns={'actual_close':'actual_open', 'next_close':'actual_close'})
		# df_bt.rename_axis('date', inplace=True)
		# df_bt.reset_index(inplace=True)
		# df_bt['actual_direction'] = np.sign(target['target']).tolist()
		# df_bt['date'] = pd.DatetimeIndex(df_bt['date']) + pd.DateOffset(hours=self.time_horizon)    


		df_bt_ = df_target_log_transformed.copy()
		df_bt_.index = pd.DatetimeIndex(df_bt_.index) + pd.DateOffset(minutes=self.time_horizon_minutes)    
		# df_bt_['target'] = np.sign(df_bt_['target'])
		# df_bt_.rename(columns={'target':'actual_direction'}, inplace=True)
		# # print(df_bt_)
		

		# print(df_bt)
		if input_str == 'backtest':
			# print(df_bt_)
			# self.df_backtest = self.pred = df_bt
			self.df_backtest = df_bt_
		elif input_str == 'rw':
			# print(df_bt)
			# self.df_realworld = self.pred_rw = df_bt
			self.df_realworld = df_bt_
			# print(self.pred_rw)
		

		# Check for inf values in columns
		inf_columns = df_train_log_transformed.columns[np.isinf(df_train_log_transformed.values).any(axis=0)]

		# if len(inf_columns) > 0:
		# 	print("The following column(s) contain inf values:")
		# 	print(inf_columns)
		# 	print(df_train_log_transformed[inf_columns])
			
		# Check for NaN values
		has_nan_or_inf = df_train_log_transformed.isna().any().any() or np.isinf(df_train_log_transformed.values).any()
		if has_nan_or_inf:
			print("There are NaN/inf values in the DataFrame.")
			exit(0)

		# print(
		return df_train_log_transformed, df_target_log_transformed
		# print(train)
		# print(target)
		# print(df_close_only)

		# if 
		
	def drop_correlated_features(self, df, threshold):
		df_copy = df.copy()

		# Remove OHCL columns
		df_ohlcv = df[["open", "high", "low", "close", "volume"]]
		df_drop = df_copy.drop(["open", "high", "low", "close", "volume"], axis=1)

		# Calculate Pierson correlation
		df_corr = df_drop.corr()

		columns = np.full((df_corr.shape[0],), True, dtype=bool)
		for i in range(df_corr.shape[0]):
			for j in range(i+1, df_corr.shape[0]):
				if df_corr.iloc[i,j] >= threshold or df_corr.iloc[i,j] <= -threshold:
					if columns[j]:
						columns[j] = False

		selected_columns = df_drop.columns[columns] ### uncorrelated columns
		df_dropped = df_drop[selected_columns] 
		# cols_to_drop = df_dropped.columns[(df_dropped < 0).any()].tolist() ### dropping columns that have negative as we will take log diff
		# df_dropped.drop(cols_to_drop, axis=1, inplace=True)


		### add open, high, low, close, and volumne cols to the uncorrelated dataframe
		df_uncorrelated = pd.concat([df_ohlcv, df_dropped], axis=1)

		# df_uncorrelated = df_dropped.assign(open=df['open'], high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
		# df_uncorrelated = df_uncorrelated.replace(0, 1)

		# print(df_uncorrelated)

		# if plot:
		#     # Plot Heatmap Correlation
		#     fig = plt.figure(figsize=(8,8))
		#     ax = sns.heatmap(df_uncorrelated.corr(), annot=True, square=True)
		#     ax.set_yticklabels(ax.get_yticklabels(), rotation=0) 
		#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
		#     fig.tight_layout()
		#     plt.show()

		return df_uncorrelated

	def create_X_y(self, tmp_df, input_str):
	
		df = tmp_df.copy()
		df = ut.rolling_window(df, self.rw_size)
		print('*'*100, 'df')
		print(df)
		df = ut.log_diff(df, [c for c in df.columns if c not in ['original_target', 'original_open']], self.rw)
		print(df)
		
		X = df[[x for x in df.columns if x not in ['target', 'original_target', 'original_open']]]
		y = df[['target']].copy()
		print('*'*100, 'X, y')
		print(X)
		print(y)

		if self.learner == 'classification':
			y = np.sign(y)
			y.loc[y.target == 0, "target"] = 1

		elif self.learner == 'drl':
			y.loc[:, 'direction'] = np.sign(y[['target']])
			# fix for any cases where target might be 0 due to forward filling in the data
			y.loc[y.direction == 0, "direction"] = 1
		
		elif self.learner == 'timeseries_darts':
			from darts import TimeSeries
			data = tmp_df.copy()
			print(data)
			df_close = pd.DataFrame()
			df_close['close'] = data['close']
			df_close['close'] = np.log(df_close['close']).diff()
			df_close = df_close.rename_axis('datetime').reset_index()
			df_close.dropna(inplace=True)
			print(df_close)
			df_close['datetime'] = pd.to_datetime(df_close.datetime).dt.tz_localize(None)
			# self.darts_train = TimeSeries.from_dataframe(df_close, time_col='datetime', value_cols='close')
			# X = y = TimeSeries.from_dataframe(df_close, time_col='datetime', value_cols='close')
			# print(self.darts_train)
		
		if input_str == 'backtest':
			# creation of prediction dataframe, containing the ground truth values
			self.pred = pd.DataFrame(y).rename(columns={'target': 'actual_direction'})
			self.pred['actual_close'] = df[['original_target']]
			self.pred['actual_open'] = df[['original_open']]
			self.pred.rename_axis('date', inplace=True)
			self.pred.reset_index(inplace=True)
			self.pred['date'] = pd.DatetimeIndex(self.pred['date']) + pd.DateOffset(hours=self.time_horizon)            
			self.pred = self.pred[['date', 'actual_open', 'actual_close', 'actual_direction']]
			# print(self.pred.shape, self.pred)

			self.create_timeseries_darts(input_str, tmp_df.copy())
			
		elif input_str == 'rw':
			# creation of prediction dataframe, containing the ground truth values
			self.pred_rw = pd.DataFrame(y).rename(columns={'target': 'actual_direction'})
			self.pred_rw['actual_close'] = df[['original_target']]
			self.pred_rw['actual_open'] = df[['original_open']]
			self.pred_rw.rename_axis('date', inplace=True)
			self.pred_rw.reset_index(inplace=True)
			self.pred_rw['date'] = pd.DatetimeIndex(self.pred_rw['date']) + pd.DateOffset(hours=self.time_horizon)
			self.pred_rw = self.pred_rw[['date', 'actual_open', 'actual_close', 'actual_direction']]
			# print(self.pred_rw.shape, self.pred_rw)

			self.create_timeseries_darts(input_str, tmp_df.copy())

		return X, y

	def process_df(self):
		self.X_train, self.y_train = self.create_X_y(self.train, 'train')
		# self.X_test, self.y_test = self.create_X_y(self.test, 'backtest')
		# self.X_test_rw, self.y_test_rw = self.create_X_y(self.rw_test, 'rw')

		# print("Length of train, backtest, realworld", len(self.X_train), len(self.X_test), len(self.X_test_rw))
		# print("Train    ", str(self.train.index[0]).split(' ')[0], 'to', str(self.train.index[-1]).split(' ')[0])
		# print("Backtest ", str(self.test.index[0]).split(' ')[0], 'to', str(self.test.index[-1]).split(' ')[0])
		# print("Realworld", str(self.rw_test.index[0]).split(' ')[0], 'to', str(self.rw_test.index[-1]).split(' ')[0])

		# print('*'*100)
		# print(self.X_train, self.y_train)
		# print('*'*100)
		# print(self.X_test, self.y_test)
		# print('*'*100)
		# print(self.X_test_rw, self.y_test_rw)


		# # # print("Test", self.test.index[0], self.test.index[-1], len(self.X_test))
		# # # print("Test", self.rw_test.index[0], self.rw_test.index[-1], len(self.X_test_rw))
	
	def add_custom_indictors(self):		
		df_data = self.data.copy()
		self.list_binary_indicators = []


		# if 'all' in self.list_indicators:
		# 	df_data = add_all_ta_features(df_data, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
		# 	# print(df_data)

		# 	### find and drop cols that have 0 or negative values in them except binary ones
		# 	list_negative_cols = []
		# 	for column in df_data.columns:
		# 		if all(df_data[column].isin([-1, 0, 1])) or\
		# 			all(df_data[column].isin([-1, 1])) or\
		# 			all(df_data[column].isin([1, 0])):
		# 			a=0

		# 		elif any(df_data[column] == 0) or any(df_data[column] < 0):
		# 			list_negative_cols.append(column)
				
			
		# 	df_data = df_data.drop(list_negative_cols, axis=1)
		# 	# print(df_data)
		# 	# print(df_data.columns)

		# 	df_data = self.drop_correlated_features(df_data, self.correlation_threshold)
		# 	# print(df_data)
		# 	print(df_data.columns)

		# else:
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

				

		# list_non_logged_cols = []
		# list_negative_cols = []
		# for column in df_data.columns:
		# 	# if column not in binary_cols and column not in negative_cols:
		# 		# df_train_log_transformed[column] = np.log(df_rolling_window[column]).diff()
		# 	if all(df_data[column].isin([-1, 0, 1])) or\
		# 		all(df_data[column].isin([-1, 1])) or\
		# 		all(df_data[column].isin([1, 0])):
		# 		list_non_logged_cols.append(column)
		# 		print('\n list_non_logged_cols')
		# 		print(df_data[column])

		# 	elif any(df_data[column] == 0) or any(df_data[column] < 0):
		# 		list_negative_cols.append(column)
		# 		print(df_data[column])

		# print(list_non_logged_cols)
		# print(list_negative_cols)
	
		# print(df_data)
		# df_data = self.drop_correlated_features(df_data, 0.5)
		df_data.dropna(inplace=True)
		self.data = df_data
		# print(df_data)
		# print(df_data.columns)

		# df_data = self.drop_correlated_features(df_data, self.correlation_threshold)
		# print(df_data)
		# print(df_data.columns)
		# print(df_data.columns)
		
