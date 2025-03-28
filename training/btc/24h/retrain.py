import torch
import os
from apollo_gamma.learners import learner_drl
import config
from apollo_gamma.learners import learner_drl
from apollo_gamma.utils import paths, Dataset, utils
from apollo_gamma.drl_envs.env_ld_sw_dataset import EnvTrading

############################################################################################
### Declare model name here that needs to be retrained (with extension, i.e., zip or pt)
############################################################################################
model_name = 'BTC_24h_sb3_sac_15_0_20Sep22_178_89.zip'
############################################################################################

def retrain():
	current_dir = os.path.dirname(os.path.realpath(__file__))
	model_path = os.path.join(current_dir, 'models', model_name)
	base_name = os.path.basename(model_path)
	extension = base_name.split('.')[-1]
	forwardtest_duration = int(base_name.split('_')[-1].split('.')[0])
	backtest_duration = base_name.split('_')[-2]
	trained_until = base_name.split('_')[-3]
	utc = base_name.split('_')[-4]
	rolling_window_size = int(base_name.split('_')[-5])
	algo = base_name.split('_')[-6]
	lib = base_name.split('_')[-7]
	str_time_horizon = base_name.split('_')[-8]
	new_model_name = base_name.split('.')[0] + '_updated' 

	split_triple = [config.training_start_date, config.backtest_duration, config.forwardtest_duration]
	instrument_exchange_timehorizon_utc = [config.instrument, config.exchange, config.str_time_horizon, config.utc]
	data_path = paths.get_hourly_utc(config.instrument, config.exchange, config.str_time_horizon, config.utc)
	obj_dataset = Dataset.Dataset(config.instrument, config.str_time_horizon, data_path, split_triple, config.list_indicators, rolling_window_size=rolling_window_size, retrain=True, trained_until=trained_until)
	arg_vector = [obj_dataset, config.str_time_horizon]
	train_env = EnvTrading('train', arg_vector)


	if extension == 'zip':
		sb_algo = utils.get_retrain_algo(algo)
		trained_model = sb_algo.load(model_path)
		device = 'cpu'
		new_model = sb_algo("MlpPolicy", train_env, verbose=0, device=device)
		arg_vector = [trained_model, train_env, obj_dataset, current_dir, new_model]
		obj_sb1 = learner_drl.SBPretraining(arg_vector)
		library = 'sb3'
		arg_vector = [sb_algo, current_dir, train_env, instrument_exchange_timehorizon_utc, library, obj_dataset, rolling_window_size, config.transaction_fee_percent, config.take_profit_percent, config.stop_loss_percent, config.leverage, config.use_gpu, obj_sb1.new_model, new_model_name]

		obj_sb1 = learner_drl.SBRetraining(arg_vector)
		obj_sb1.train()

	elif extension == 'pt':
		trained_model = torch.jit.load(model_path)
		library = 'd3rlpy'
		d3_algo = utils.get_retrain_algo(algo)
		arg_vector = [d3_algo, current_dir, train_env, instrument_exchange_timehorizon_utc, library, trained_model, obj_dataset, rolling_window_size, config.transaction_fee_percent, config.take_profit_percent, config.stop_loss_percent, config.leverage, config.use_gpu, new_model_name]

		obj_d3 = learner_drl.D3Retraining(arg_vector)
		obj_d3.train()


def main():
	retrain()

if __name__ == "__main__":
	main()