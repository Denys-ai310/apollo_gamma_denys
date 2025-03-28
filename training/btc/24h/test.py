# import config
from stable_baselines3 import PPO, TD3, DDPG,  SAC, A2C
from apollo_gamma.learners import learner_drl
from apollo_gamma.utils import paths, Dataset, utils
from apollo_gamma.drl_envs.env_ld_sw_dataset import EnvTrading
import os
import torch
import sys
import numpy as np
import random
import pandas as pd
from apollo_gamma.backtest import backtest
import quantstats as qs
import glob
# set random seeds in random module, numpy module and PyTorch module.
seed = 818
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# torch.manual_seed(torch.initial_seed())




def perform_backtest(input_data, df_backtest, model, lib):

	if lib == 'sb3':
		# print(input_data, tmp_model)
		input_data = np.array(input_data, dtype= np.float32)
		model_predictions = np.ravel(model.predict(input_data, deterministic=True)[0])
	elif lib == 'd3rlpy':
		obs = np.array(input_data, dtype= np.float32)
		obs = torch.from_numpy(obs)
		model_predictions = np.ravel(model(obs)) 


	df_backtest['predicted_direction'] = np.sign(model_predictions)
	df_pred = df_backtest.dropna()
	predictions = df_pred

	obj_backtest = backtest.Backtest(predictions, instrument_exchange_timehorizon_utc, transaction_fee=0.0)
	ledger, ending_balance, sharpe, r2, pnl_percent = obj_backtest.run()
	r2 = round(r2, 2)

	# print(ledger)
	print(ending_balance, sharpe, r2, pnl_percent)

	

	return ledger, ending_balance, sharpe, r2, pnl_percent


def load_custom_model(model_path, lib, algo):
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
	# elif lib == 'ML':
	# 	model = joblib.load(model_path)

	return model



#################################################################################################
### set these parameters before training
#################################################################################################
instrument = 'btc'                       # btc/eth/xrp (the data should be downloaded first)
exchange = 'dydx'                        # bitmex/binance/derabits (atm dydx is supported)
str_time_horizon = '24h'                 # 30min, 1h, 2h, 3h, 4h, 6h, 8h, 12h, 24h 
utc = 0

training_start_date = '2013-01-01'       # must be greator than 2012-01-01
backtest_duration = 180                  # in days
forwardtest_duration = 90                # in days

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

### 24h
list_indicators = ['vwma', 'rsi', 'bollinger_bands_binary', 'engulfing_candle_binary', 
                    'obv_divergence_binary', 'volatility_kcli', 'trend_mass_index',  
                    'trend_aroon_down', 'trend_psar_up_indicator', 'trend_psar_down_indicator' ]
						
 

######################################################################################################





current_dir = os.path.dirname(os.path.realpath(__file__))
models_directory = os.path.join(current_dir, 'models')

list_of_models = glob.glob(models_directory + '/*')
print(list_of_models)

for model_path in list_of_models:
	# base_name = 'BTC_24h_d3rlpy_bear_29.pt'
	base_name = os.path.basename(model_path)
	print(base_name)

	if '.html' in base_name:
		continue


	lib = base_name.split('_')[-3]
	algo = base_name.split('_')[-2]
	rolling_window_size = int(base_name.split('_')[-1].split('.')[0])
	
	if lib == 'd3rlpy':
		model_name_prefix = base_name.replace('.pt', '')
	else:
		model_name_prefix = base_name.replace('.zip', '')

	model = load_custom_model(model_path, lib, algo)

	# path_model = os.path.join(current_dir, 'models', 'BTC_24h_d3rlpy_bear_29.pt')


	table_name = f'{instrument}_{str_time_horizon}'
	path_bt_db = os.path.join(current_dir, 'backtest', 'db', 'backtest.db') 
	path_ft_db = os.path.join(current_dir, 'forwardtest', 'db', 'forwardtest.db') 
	path_bt_results = os.path.join(current_dir, 'backtest', 'results') 
	path_ft_results = os.path.join(current_dir, 'forwardtest', 'results') 

	

	split_triple = [training_start_date, backtest_duration, forwardtest_duration]
	instrument_exchange_timehorizon_utc = [instrument, exchange, str_time_horizon, utc]
	data_path = paths.get_hourly_utc(instrument, exchange, str_time_horizon, utc)
	# rolling_window_size = 29

	obj_dataset = Dataset.Dataset(instrument, str_time_horizon, data_path, split_triple, list_indicators, rolling_window_size)

	#########################################################

	input_data = pd.concat([obj_dataset.X_backtest, obj_dataset.X_forwardtest])
	df_backtest = pd.concat([obj_dataset.df_backtest, obj_dataset.df_forwardtest])
	df_best_ledger, ending_balance, sharpe, r2, pnl_percent = perform_backtest(input_data, df_backtest, model, lib)
	# print(df_best_ledger)

	

	df_best_ledger.set_index('datetime', inplace=True)
	df_best_ledger = df_best_ledger[df_best_ledger['sell_price'] != 0]
	df_best_ledger['pnl'] = df_best_ledger['pnl'] + transaction_fee_percent
	df_best_ledger = df_best_ledger['pnl']/100
	df_best_ledger.index = df_best_ledger.index.tz_convert(None)

	path_report = os.path.join(models_directory, f'{model_name_prefix}.html')
	title_report = str(base_name).split('_')[-2].upper() + ' Results'
	qs.reports.html(df_best_ledger, title=title_report, output=True, compounded=False, download_filename=path_report)



	input_data = obj_dataset.X_backtest
	df_backtest = obj_dataset.df_backtest
	df_best_ledger, ending_balance, sharpe, r2, pnl_percent = perform_backtest(input_data, df_backtest, model, lib)

	bt_record = [model_name_prefix, ending_balance, pnl_percent, r2, sharpe, utils.get_ledger_days(df_best_ledger)]
	utils.insert_db_drl(path_bt_db, table_name, [bt_record])

	
	df_best_ledger.to_csv(os.path.join(path_bt_results, f'bt_ledger_{model_name_prefix}.csv'))

	input_data = obj_dataset.X_forwardtest
	df_backtest = obj_dataset.df_forwardtest
	df_best_ledger, ending_balance, sharpe, r2, pnl_percent = perform_backtest(input_data, df_backtest, model, lib)

	bt_record = [model_name_prefix, ending_balance, pnl_percent, r2, sharpe, utils.get_ledger_days(df_best_ledger)]
	utils.insert_db_drl(path_ft_db, table_name, [bt_record])
	df_best_ledger.to_csv(os.path.join(path_ft_results, f'ft_ledger_{model_name_prefix}.csv'))





