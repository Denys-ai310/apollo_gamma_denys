import config
from stable_baselines3 import PPO, TD3, DDPG,  SAC, A2C
from apollo_gamma.learners import learner_drl
from apollo_gamma.utils import paths, Dataset, utils
from apollo_gamma.drl_envs.env_ld_sw_dataset import EnvTrading
import os
import sys


def get_algo(algo):
	if algo == 'a2c':
		sb_algo = A2C
	elif algo == 'ppo':
		sb_algo = PPO
	elif algo == 'ddpg':
		sb_algo = DDPG
	elif algo == 'sac':
		sb_algo = SAC
	elif algo == 'td3':
		sb_algo = TD3
	else:
		print('Algorithm not found. Please check the spelling of the algorithm name.')
		exit(0)
	
	return sb_algo

def train(algo):
	current_dir = os.path.dirname(os.path.realpath(__file__))
	utils.create_results_directories(current_dir)
	sb_algo = get_algo(algo)
	split_triple = [config.training_start_date, config.backtest_duration, config.forwardtest_duration]
	instrument_exchange_timehorizon_utc = [config.instrument, config.exchange, config.str_time_horizon, config.utc]
	data_path = paths.get_hourly_utc(config.instrument, config.exchange, config.str_time_horizon, config.utc)
	rolling_window_length = 32

	
	
	library = 'sb3'
	algo_name = library + '_' + str(sb_algo).split('.')[-2]
	
	print('\n')
	print('-'*60)
	print('DRL Algorithm:', str(sb_algo).split('.')[-2].upper(), '| Instrument:', instrument_exchange_timehorizon_utc[0])
	print('-'*60)
	print('Training started...')


	for rolling_window_size in range(1, rolling_window_length+1):
		### create dataset object
		obj_dataset = Dataset.Dataset(config.instrument, config.str_time_horizon, data_path, split_triple, config.list_indicators, rolling_window_size)

		### DRL environment creation 
		arg_vector = [obj_dataset, config.str_time_horizon]
		train_env = EnvTrading('train', arg_vector)

		arg_vector = [sb_algo, current_dir, train_env, instrument_exchange_timehorizon_utc, library, obj_dataset, rolling_window_length, config.transaction_fee_percent, config.take_profit_percent, config.stop_loss_percent, config.leverage, config.use_gpu]
		
		obj_sb1 = learner_drl.CustomSB(arg_vector)
		obj_sb1.train()

		if obj_sb1.best_model:
			break

	print('-'*60)
	print('Training completed.')
	print('-'*60)


def main():
	# train(sys.argv[1])
	train('a2c')

if __name__ == "__main__":
	main()