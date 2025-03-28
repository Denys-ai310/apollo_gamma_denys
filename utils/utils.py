import logging
import warnings
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import time
import os
import shutil
import torch
import joblib
import configparser
import warnings
import sqlite3
from datetime import datetime, timedelta
import os
import sys
# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils import Dataset, paths
from backtest import backtest
import quantstats as qs
import glob 
import torch
import d3rlpy
import random
import pytz

warnings.simplefilter(action='ignore', category=FutureWarning)
config = configparser.ConfigParser(inline_comment_prefixes=";")
config_path = os.path.join(paths.root_dir, 'config.ini')

def get_config_file():
	try:
		config.read(config_path)
		return config
	except:
		print("Error in reading config file...")
		exit()

config = get_config_file()
seed = 818
# set_seed = True if (config['training']['set_seed']).lower() == 'true' else False

backtest_results_directory = 'backtest/results'
backtest_db_directory = 'backtest/db'
forwardtest_results_directory = 'forwardtest/results'
forwardtest_db_directory = 'forwardtest/db'
models_directory = 'models'

# backtest_results_directory = str(config['training']['backtest_results_directory'])
# backtest_db_directory = str(config['training']['backtest_db_directory'])
# forwardtest_results_directory = str(config['training']['forwardtest_results_directory'])
# forwardtest_db_directory = str(config['training']['forwardtest_db_directory'])
# models_directory = str(config['training']['models_directory'])

def set_all_seeds(set_seed=False):
	if set_seed: 
		import torch
		import numpy as np
		import random
		torch.manual_seed(seed)
		np.random.seed(seed)
		random.seed(seed)
		print(f' --------------------- Seed {seed} is set --------------------- ')
	
	return set_seed, seed

def split_train_test_rw(data, split_triple, rw_size):
	training_start_date, backtest_duration, rw_test_duration = split_triple[0], split_triple[1], split_triple[2]

	rw_end = pd.to_datetime(data.index[-1])
	rw_start = pd.to_datetime(data.index[-1]) - pd.DateOffset(days = rw_test_duration)
	rw_start = rw_start - pd.DateOffset(minutes= rw_size)

	# bt_end = rw_start - pd.DateOffset(minutes = 1) + pd.DateOffset(minutes = rw_size)
	bt_end = rw_start + pd.DateOffset(minutes = rw_size)
	bt_start = bt_end - pd.DateOffset(days = backtest_duration) + pd.DateOffset(minutes = 1)
	bt_start = bt_start - pd.DateOffset(minutes = rw_size)

	train_end = bt_start - pd.DateOffset(minutes = 1)
	train_start = pd.to_datetime(training_start_date, utc=True)
	# print(data.index == train_start)
	# print(train_start, train_end)

	# print(data.index)
	# data.index = pd.to_datetime(data.index)
	# if (data.index == train_start).any():
	if train_start not in data.index:
		train_start = pd.to_datetime(data.index[0])

	# print(rw_start, rw_end)
	# print(bt_start, bt_end)
	# print(train_start, train_end)

	# train = data[ data.index.between(train_start, train_end)]
	train = data[(data.index >= train_start) & (data.index <= train_end)]
	bt = data[(data.index >= bt_start) & (data.index <= bt_end)]
	rw = data[(data.index >= rw_start) & (data.index <= rw_end)]
	# print(train)
	# print(bt)
	# print(rw)
	
	return train, bt, rw

def split_train_test_rw_retrain(data, split_triple, rw_size, trained_until):
	training_start_date, backtest_duration, rw_test_duration = split_triple[0], split_triple[1], split_triple[2]

	train_start = pd.to_datetime(training_start_date, utc=True)
	train_end = datetime.strptime(trained_until, '%d%b%y')  
	train_end = train_end.replace(tzinfo=pytz.UTC)
	train = data[(data.index >= train_start) & (data.index <= train_end)]
	# print(train)

	bt_start = pd.to_datetime(train.index[-1])
	bt_end = bt_start + pd.DateOffset(days = backtest_duration)
	bt_start = bt_start - pd.DateOffset(minutes = rw_size)

	# rw_start = bt_end



	# rw_end = pd.to_datetime(data.index[-1])
	# rw_start = pd.to_datetime(data.index[-1]) - pd.DateOffset(days = rw_test_duration)
	# rw_start = rw_start - pd.DateOffset(minutes= rw_size)

	# # bt_end = rw_start - pd.DateOffset(minutes = 1) + pd.DateOffset(minutes = rw_size)
	# bt_end = rw_start + pd.DateOffset(minutes = rw_size)
	# bt_start = bt_end - pd.DateOffset(days = backtest_duration) + pd.DateOffset(minutes = 1)
	# bt_start = bt_start - pd.DateOffset(minutes = rw_size)

	# train_end = bt_start - pd.DateOffset(minutes = 1)
	# train_start = pd.to_datetime(training_start_date, utc=True)
	# # print(data.index == train_start)
	# # print(train_start, train_end)

	# # print(data.index)
	# # data.index = pd.to_datetime(data.index)
	# # if (data.index == train_start).any():
	# if train_start not in data.index:
	# 	train_start = pd.to_datetime(data.index[0])

	# # print(rw_start, rw_end)
	# # print(bt_start, bt_end)
	# # print(train_start, train_end)

	# train = data[ data.index.between(train_start, train_end)]
	# train = data[(data.index >= train_start) & (data.index <= train_end)]
	bt = data[(data.index >= bt_start) & (data.index <= bt_end)]
	rw = data[data.index > bt_end]
	# print(train)
	# print(bt)
	# print(rw)

	rw_start = pd.to_datetime(rw.index[0])
	rw_end = rw_start + pd.DateOffset(days = rw_test_duration)
	new_training_data = data[data.index > rw_end]
	# print(new_training_data)

	train = pd.concat([train, new_training_data])
	# print(train)
	
	return train, bt, rw

def create_rolling_window(df, rw_size):
	input_features = []
	input_dates = []
	target = []
	close = []

	### loop will run for n-1 entries where last entry is used for the target
	# print(df.shape[0], rw_size)
	for i in range (df.shape[0] - rw_size):
		# print(i, i+rw_size)
		upto_current_index = i+rw_size-1
		next_index = i+rw_size

		### input features with rolling window
		x = df.values[i : next_index]
		input_features.append(x.ravel().tolist())
		input_dates.append(df.index[upto_current_index])

		### target for input features
		y = df[['close']].values[next_index]
		target.append(y.ravel().tolist())

		### open and close for backtest df
		z = df[['close']].values[upto_current_index]
		close.append(z.ravel().tolist())


	df = pd.DataFrame(input_features, index=input_dates)
	# print(df)
	# print(len(df), len(close))
	df[['actual_close']] = close
	df[['next_close']] = target

	return df

def df_log_difference(df):
	# print(df)

	# ohlcv_names = [0, 1, 2, 3, 4]

	for c in df.columns:
		if (c == 'actual_close') or (c == 'next_close'):
			df[c+'_log_diff'] = np.log(df[c]).diff()
		else:
			# if c in ohlcv_names:
				# print(c, type(c))
			df[c] = np.log(df[c]).diff()
	df = df.dropna()
	# print(df)
	return df

def np_create_rolling_window(df, rw_size):
	
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

def create_results_directories(current_dir):
	create_if_not_exists(os.path.join(current_dir, backtest_results_directory))
	create_if_not_exists(os.path.join(current_dir, backtest_db_directory))
	create_if_not_exists(os.path.join(current_dir, forwardtest_results_directory))
	create_if_not_exists(os.path.join(current_dir, forwardtest_db_directory))
	create_if_not_exists(os.path.join(current_dir, models_directory))

	# create_override_dir(os.path.join(current_dir, backtest_results_directory))
	# create_override_dir(os.path.join(current_dir, backtest_db_directory))
	# create_override_dir(os.path.join(current_dir, forwardtest_results_directory))
	# create_override_dir(os.path.join(current_dir, forwardtest_db_directory))
	# create_override_dir(os.path.join(current_dir, models_directory))

def create_override_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)
	else:
		shutil.rmtree(path)
		os.makedirs(path)
	return path

def create_if_not_exists(path):
	if not os.path.exists(path):
		try:
			os.makedirs(path)  
		except:
			a=0

def delete_dir(path):
	if os.path.exists(path) and os.path.isdir(path):
		shutil.rmtree(path)

def delete_file(path):
	if os.path.exists(path):
		os.remove(path)

def check_status_db(status_db_path, learner):
	if os.path.exists(status_db_path):
		conn = sqlite3.connect(status_db_path)
		c = conn.cursor()

		### if db is empty
		c.execute('SELECT name FROM sqlite_master')
		result=c.fetchall()
		if result == []:
			return False

		### check if learner is already trained
		tmp_df = pd.read_sql_query("SELECT * from status", conn)
		bool_learner = tmp_df.applymap(lambda x: x == learner).any().any()
		return bool_learner
	else:
		return False
		
def update_status_db_ml(status_db_path, learner):
	conn = sqlite3.connect(status_db_path)
	c = conn.cursor()
	df = pd.DataFrame([[learner, 'completed']], columns=['learner', 'status'])
	df.to_sql('status', conn,if_exists='append', index=False)               
	conn.commit()
	conn.close()

def get_paths_drl(current_dir, model_name):
	path_bt_results, path_bt_db, path_ft_results, path_ft_db, path_for_saving_models = get_model_paths(current_dir, model_name)

	# create_override_dir(path_bt_results)
	# delete_file(path_bt_db)
	# create_override_dir(path_ft_results)
	# delete_file(path_ft_db)
	# create_override_dir(path_for_saving_models)

	return path_bt_results, path_bt_db, path_ft_results, path_ft_db, path_for_saving_models	

def get_model_paths(current_dir, model_name):    
	return os.path.join(current_dir, f'{backtest_results_directory}'),\
			os.path.join(current_dir, f'{backtest_db_directory}/backtest.db'),\
			os.path.join(current_dir, f'{forwardtest_results_directory}'),\
			os.path.join(current_dir, f'{forwardtest_db_directory}/forwardtest.db'),\
			os.path.join(current_dir, f'{models_directory}/{model_name}')

# def predict_and_backtest_drl(input_data, model, *args):
# 	# Ensure input_data is the correct shape
# 	if isinstance(input_data, pd.DataFrame):
# 		input_data = input_data.to_numpy()
	
# 	if len(input_data.shape) == 2:
# 		if input_data.shape[1] > 13:
# 			input_data = input_data[:, :13]
# 		# If predicting one at a time
# 		predictions = []
# 		for obs in input_data:
# 			pred = model.predict(obs.reshape(-1), deterministic=True)[0]
# 			predictions.append(pred)
# 		model_predictions = np.ravel(predictions)
# 	else:
# 		# If input is already single observation
# 		model_predictions = np.ravel(model.predict(input_data, deterministic=True)[0])
	
# 	df_pred = pd.DataFrame({'predicted_close': model_predictions})
# 	df_pred['predicted_direction'] = np.sign(model_predictions)
# 	df_pred = df_pred.dropna()
# 	predictions = df_pred
	
# 	obj_backtest = backtest.Backtest(predictions, *args, take_profit=0.01, stop_loss=0.01, transaction_fee=0.001, leverage=1.0)
# 	ledger, ending_balance, sharpe, r2, pnl_percent = obj_backtest.run()
# 	r2 = round(r2, 2)

# 	return ledger, ending_balance, sharpe, r2, pnl_percent, df_pred

def predict_and_backtest_drl(library, bt, tmp_model, input_data, df_pred, instrument_exchange_timehorizon_utc, transaction_fee, take_profit_percent, stop_loss_percent, leverage):
	if library == 'sb1' or library == 'sb3':
		input_data = np.array(input_data, dtype= np.float32)
		model_predictions = np.ravel(tmp_model.predict(input_data, deterministic=True)[0])
	elif library == 'd3rlpy':
		
		
		# # set random seeds in random module, numpy module and PyTorch module.
		seed = 818
		# # d3rlpy.seed(seed)
		torch.manual_seed(seed)
		# np.random.seed(seed)
		# random.seed(seed)

		obs = np.array(input_data, dtype= np.float32)
		obs = torch.from_numpy(obs)
		model_predictions = np.ravel(tmp_model.predict(obs))  

	df_pred['predicted_close'] = model_predictions
	df_pred['predicted_direction'] = np.sign(model_predictions)
	df_pred = df_pred.dropna()
	predictions = df_pred
	
	obj_backtest = backtest.Backtest(predictions, instrument_exchange_timehorizon_utc, take_profit=take_profit_percent, stop_loss=stop_loss_percent, transaction_fee=transaction_fee, leverage=leverage)
	ledger, ending_balance, sharpe, r2, pnl_percent = obj_backtest.run()
	r2 = round(r2, 2)

	# torch.manual_seed(torch.initial_seed())

	return ledger, ending_balance, sharpe, r2, pnl_percent, df_pred

def get_accuracies(df_predictions, df_ledger, transaction_fee):

	value_counts = df_predictions['predicted_direction'].value_counts()

	if (df_predictions['predicted_direction'].isin([-1]).any()) and (df_predictions['predicted_direction'].isin([1]).any()):
		total_predictions = int(value_counts[1]+value_counts[-1])
		total_long_predictions = value_counts[1]
		total_short_predictions = value_counts[-1]
		######################################################################################################
		df_sell_only = df_ledger.loc[df_ledger['action'].str.contains('sell')]
		df_sell_only['pnl'] = df_sell_only['pnl'] + -transaction_fee

		total_trades = len(df_sell_only)
		total_long_trades = df_sell_only.loc[df_sell_only['predicted_direction'] == 'short'].shape[0]
		total_short_trades = df_sell_only.loc[df_sell_only['predicted_direction'] == 'long'].shape[0]
		
		correct_long_trades = df_sell_only[(df_sell_only['predicted_direction'] == 'short') & (df_sell_only['pnl'] > 0)].shape[0]
		correct_short_trades = df_sell_only[(df_sell_only['predicted_direction'] == 'long') & (df_sell_only['pnl'] > 0)].shape[0]
		try:
			return total_long_predictions/total_predictions, total_short_predictions/total_predictions, (correct_long_trades+correct_short_trades)/total_trades, correct_long_trades/total_long_trades, correct_short_trades/total_short_trades
		except:
			return 0, 0, 0, 0, 0
	else:
		return 0, 0, 0, 0, 0
	
def insert_db_drl(path_db, table_name, records):
	conn = sqlite3.connect(path_db)
	c = conn.cursor()
	# c.execute(f"DROP TABLE IF EXISTS {table_name}")
	c.execute(f""" CREATE TABLE IF NOT EXISTS 
	\"{table_name}\"(
		model_name DATATYPE,
		balance DATATYPE,
		pnl_percent DATATYPE,
		r2 DATATYPE,
		sharpe DATATYPE,
		duration_in_days DATATYPE
	)
	""")
	conn.commit()
	c.executemany(f"""INSERT INTO \"{table_name}\"
					VALUES(?,?,?,?,?,?);""",records)
	conn.commit()
	conn.close()

def delete_from_db(path_db, table_name, algo_name):
	if os.path.exists(path_db):
		conn = sqlite3.connect(path_db)
		c = conn.cursor()

		### check if table exists
		c.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
		table_exists = c.fetchone() is not None

		if table_exists:
			### if db is empty
			c.execute('SELECT name FROM sqlite_master')
			result=c.fetchall()
			if len(result) > 0:
				tmp_df = pd.read_sql_query(f"SELECT * from {table_name}", conn)
				new_df = tmp_df[~tmp_df['model_name'].str.contains(algo_name)]
				
				c = conn.cursor()
				new_df.to_sql(table_name, conn, if_exists='replace', index=False)               
				conn.commit()
				conn.close()


def delete_exisiting_data(algo_name, table_name, path_bt_results, path_bt_db, path_ft_results, path_ft_db, models_directory):
	# print(algo_name)
	bt_file = glob.glob(os.path.join(path_bt_results, f'*{algo_name}*'))
	if len(bt_file) > 0:
		delete_file(bt_file[0])

	ft_file = glob.glob(os.path.join(path_ft_results, f'*{algo_name}*'))
	if len(ft_file) > 0:
		delete_file(ft_file[0])
	
	best_file = glob.glob(os.path.join(models_directory, f'*{algo_name}*'))
	if len(best_file) > 0:
		if os.path.exists(best_file[0]) and os.path.isdir(best_file[0]):
			shutil.rmtree(best_file[0])
		else:
			os.path.exists(best_file[0])
			os.remove(best_file[0])
		# delete_file(best_file[0])

	delete_from_db(path_bt_db, table_name, algo_name)
	delete_from_db(path_ft_db, table_name, algo_name)


def get_ledger_days(ledger):

	ledger['datetime'] = pd.to_datetime(ledger['datetime'])
	start_date = ledger['datetime'].min()
	end_date = ledger['datetime'].max()

	# Calculate the number of days between start and end dates
	return int((end_date - start_date).days)

def min_max_normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)


def get_retrain_algo(algo):
	from stable_baselines3 import PPO, TD3, DDPG,  SAC, A2C
	from d3rlpy.algos import PLAS, BEAR, AWAC, TD3PlusBC, CQL
	if algo == 'bear':
		return BEAR
	elif algo == 'awac':
		return AWAC
	elif algo == 'td3plusbc':
		return TD3PlusBC
	elif algo == 'cql':
		return CQL
	elif algo == 'plas':
		return PLAS
	if algo == 'ddpg':
		return DDPG
	elif algo == 'ppo':
		return PPO
	elif algo == 'td3':
		return TD3
	elif algo == 'sac':
		return SAC
	elif algo == 'a2c':
		return A2C
	else:
		print('Algorithm not found. Please check the spelling of the algorithm name.')
		exit(0)