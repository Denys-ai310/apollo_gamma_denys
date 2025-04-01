import config
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from d3rlpy.algos import PLAS, BEAR, AWAC, TD3PlusBC, CQL
# Add project root to Python path
from app.learners import learner_drl
from app.utils import paths, Dataset, utils, d3_dd_dataset
from app.drl_envs.env_ld_sw_dataset import EnvTrading

import warnings
# import time
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import torch
import numpy as np
import random
# torch.set_warn_always(False)

# set random seeds in random module, numpy module and PyTorch module.
seed = 818
# d3rlpy.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def get_algo(algo):
	if algo == 'bear':
		d3_algo = BEAR
	elif algo == 'awac':
		d3_algo = AWAC
	elif algo == 'td3plusbc':
		d3_algo = TD3PlusBC
	elif algo == 'cql':
		d3_algo = CQL
	elif algo == 'plas':
		d3_algo = PLAS
	else:
		print('Algorithm not found. Please check the spelling of the algorithm name.')
		exit(0)
	
	return d3_algo

def train(algo):
	current_dir = os.path.dirname(os.path.realpath(__file__))
	utils.create_results_directories(current_dir)
	# train_env = env_configuratoin.train_env
	# backtest_env = env_configuratoin.train_env
	# realworld_env = env_configuratoin.train_env
	# bt_env_d3 = env_configuratoin.train_env
	# instrument_exchange_timehorizon_utc = env_configuratoin.instrument_exchange_timehorizon_utc
	# obj_dataset = env_configuratoin.obj_dataset


	d3_algo = get_algo(algo)
	library = 'd3rlpy'
	algo_name = library + '_' + str(d3_algo).split('.')[-2]

	split_triple = [config.training_start_date, config.backtest_duration, config.forwardtest_duration]
	instrument_exchange_timehorizon_utc = [config.instrument, config.exchange, config.str_time_horizon, config.utc]
	data_path = paths.get_hourly_utc(config.instrument, config.exchange, config.str_time_horizon, config.utc)
	rolling_window_length = 2

	print('\n')
	print('-'*60)
	print('DRL Algorithm:', str(d3_algo).split('.')[-2].upper(), '| Instrument:', instrument_exchange_timehorizon_utc[0])
	print('-'*60)
	print('Training started...')


	for rolling_window_size in range(1, rolling_window_length+1):

		### create dataset object
		obj_dataset = Dataset.Dataset(config.instrument, config.str_time_horizon, data_path, split_triple, config.list_indicators, rolling_window_size)

		### DRL environment creation 
		arg_vector = [obj_dataset, config.str_time_horizon]
		train_env = EnvTrading('train', arg_vector)

		# ### create dataset
		# print('Creating datadriven dataset...')
		dataset_path = os.path.join(current_dir, 'dd_dataset')
		utils.create_if_not_exists(dataset_path)
		dataset_path = os.path.join(current_dir, 'dd_dataset', f'{config.str_time_horizon}_{algo}_dataset.h5')
		d3_dd_dataset.create(train_env, dataset_path)
			
		# for i in range(len(algo_array)):
		
		# print('\n' + library + '_' + str(algo_array[i]).split('.')[-2])
		arg_vector = [d3_algo, current_dir, train_env, instrument_exchange_timehorizon_utc, library, dataset_path, obj_dataset, rolling_window_length, config.transaction_fee_percent, config.take_profit_percent, config.stop_loss_percent, config.leverage, config.use_gpu]

		obj_d3 = learner_drl.CustomD3RLPY(arg_vector)
		obj_d3.train()

		if obj_d3.best_model:
			break

	print('-'*60)
	print('Training completed.')
	print('-'*60)

def main():
	if len(sys.argv) > 1:
		train(sys.argv[1])
	else:
		train('bear')  # Default to A2C if no argument provided

if __name__ == "__main__":
	main()