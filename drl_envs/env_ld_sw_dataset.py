import gym
import numpy as np
import pandas as pd 
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score  
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
# from utils import paths
np.seterr(invalid='raise')

class EnvTrading(gym.Env):
	def __init__(self, input_str, argv):
		self.input_str = input_str
		self.obj_dataset = argv[0]

		self.str_time_horizon = argv[1]
		self.time_horizon_minutes = (int(self.str_time_horizon.split('h')[0]) * 60 ) if 'h' in self.str_time_horizon else int(self.str_time_horizon.split('min')[0]) 

		self.episode_length = 100
		self.reward_function = 'error'
		self.action_space_str = 'continuous' 

		### convert train and label dataframes to numpy for faster execution
		self.np_conversions()

		### load time horizon data for reward calculation etc
		self.load_time_horizon_data()	

		self.directional_reward = 0.01

		self.training_length = self.np_train.shape[0]
		self.episode_timestep = 0
		self.number_of_episodes = 0			
		self.reset_global_timestep()

		state_space_length = self.np_train.shape[1] 
		# print(self.np_train.shape)

		self.observation_space = gym.spaces.Box(low= -1e6, high=1e4, shape=(state_space_length,), dtype=np.float32)
		if self.action_space_str == 'continuous':
			self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
		elif self.action_space_str == 'discrete':
			self.action_space = gym.spaces.Discrete(2)

		self.action_range = [-1, 1]
		self.action_log_diff_lb_ub = []
			
	def reset_global_timestep(self):
		if self.input_str == 'train':
			self.training_dataset_length_timesteps = self.training_length
			self.global_timestep = 0
			self.global_timestep = random.randint(0, int(self.training_length/2)) 
			self.done_true_condition = self.training_length
		# elif self.input_str == 'test':
		else:
			self.training_dataset_length_timesteps = self.training_length
			self.global_timestep = 0
			self.done_true_condition = self.training_length
		
	def rescale_actions(self, x, srcRange, dstRange):
		return (x-srcRange[0])*(dstRange[1]-dstRange[0])/(srcRange[1]-srcRange[0])+dstRange[0]

	def reset(self):

		self.global_timestep__ = self.global_timestep
		self.actual_direction = []
		self.predicted_direction = []
		self.episode_rewards = []

		self.create_state()
		self.episode_timestep = 1
		self.global_timestep += 1

		return  self.normalised_state

	def next_state(self):
		self.create_state()
		self.episode_timestep += 1
		self.global_timestep += 1

	def step(self, action):
		# print('step', self.global_timestep, action)

		### convert action from list 
		if isinstance(action, (list, tuple, np.ndarray)):
			action = action[0]
			# ### rescale action from [-1,1] to log diff range
			# action = self.rescale_actions(action, self.action_range, self.action_log_diff_lb_ub) 
			# # print(action)

		### prediction of timestep
		current_timestep = self.global_timestep-1
		next_timestep = self.global_timestep

		current_close = self.np_ohlcv[current_timestep][self.np_ohlcv_close_index]
		next_close = self.np_ohlcv[next_timestep][self.np_ohlcv_close_index]

		### actual open/close/direction for next time horizon
		actual_direction = self.np_labels[current_timestep][self.labels_direction_index]
		actual_close_log_diff = self.np_labels[current_timestep][self.labels_target_index]
		# actual_open = self.P_open[next_timestep]
		# actual_close = self.P_close[next_timestep]
		

		# print(actual_direciton, actual_close_log_diff, actual_open, actual_close, self.date[self.global_timestep])
		# print(self.np_train[0][self.train_date_index], self.np_train[-1][self.train_date_index])
		# print(self.date)
		
		### predicted direction/price for next time horizon
		predicted_close_log_diff = action
		predicted_next_close = np.exp(predicted_close_log_diff + np.log(current_close) )
		# predicted_next_close = predicted_close_log_diff * self.P_close[current_timestep] + self.P_close[current_timestep]
		
		predicted_direction = np.sign(predicted_close_log_diff)
		
		if self.action_space_str == 'discrete':
			predicted_direction = 1 if action == 1 else -1

		### error between actual and predicted price
		reward_error =  mean_absolute_error([actual_close_log_diff], [predicted_close_log_diff]) * -1
		# print(reward_error)

		if predicted_direction == actual_direction:
			reward_error = 0.001

		reward_direction = self.directional_reward if predicted_direction == actual_direction else -self.directional_reward

		### lists for accuracy calculations
		self.actual_direction.append(actual_direction)
		self.predicted_direction.append(predicted_direction)

		# ### to create backtest csv
		# if self.input_str == 'test' or self.input_str == 'realworld_bt' or self.input_str == 'future_backtest':
		# 	### used global_timestep because the current prediction is for next time horizon and BT will run on that
		# 	return_array = [
		# 						self.date[self.global_timestep],  
		# 						actual_open, \
		# 						actual_close, \
		# 						actual_direciton,
		# 						predicted_direction,\
		# 						predicted_next_close
		# 					]	
		# 	# print(self.date[self.global_timestep], actual_open, actual_close, actual_direciton, predicted_direction, predicted_next_close)
		# 	# return_array = predicted_direction
		
		# # print(self.date[self.global_timestep],  
		# # 						actual_open, \
		# # 						actual_close, \
		# # 						actual_direciton,
		# # 						predicted_direction,\
		# # 						predicted_next_close)

		#######################################################################
		### reward 
		#######################################################################
		reward = reward_error
		# reward = reward_error + reward_direction
		self.episode_rewards.append(reward)
		# print(self.global_timestep, action, reward)
		# print(self.normalised_state)

		#######################################################################
		#### Reset global timestep - increment episodes
		#######################################################################	
		done = False
		if self.global_timestep >= self.done_true_condition-1:
			self.reset_global_timestep()
			done = True
			self.number_of_episodes += 1

		if self.input_str == 'train' or self.input_str == 'eval':
			if self.global_timestep % self.episode_length == 0:
				self.number_of_episodes += 1
				done = True
		
		if done:
			# if self.number_of_episodes % 100 == 0:
				# print("*"*65)
				# print ("Episode completed: ", self.number_of_episodes)
				# print ("Reward ", round(sum (self.episode_rewards), 2),  )
				# self.directional_accuracy_()
			self.episode_rewards = []

		self.next_state()


		return self.normalised_state, reward, done, {} 

		# if self.input_str == 'test' or self.input_str == 'realworld_bt' or self.input_str =='future_backtest':
		# 	save_model = False
		# 	if done:
		# 		# save_model =  self.directional_accuracy_sqj()
		# 		save_model = True				
		# 	return self.normalised_state, reward, done, return_array, save_model
		# elif self.input_str == 'optimise':
		# 	return self.normalised_state, reward, done, {} 
		# else:
		# 	return self.normalised_state, reward, done, {} 

	def directional_accuracy_(self):
		y_true = self.actual_direction
		y_pred = self.predicted_direction

		correct_positive_trend_pred = 0
		correct_negative_trend_pred = 0
		total_positive = 0
		total_negative = 0
		for i in range(len(y_true)):

			if (y_true[i] == 1 and y_pred[i] == 1):
				correct_positive_trend_pred = correct_positive_trend_pred + 1
			elif (y_true[i] == -1 and y_pred[i] == -1):
				correct_negative_trend_pred = correct_negative_trend_pred + 1
			
			if (y_true[i] == 1):
				total_positive += 1
			else:
				total_negative += 1
		
		# print("*"*30, self.input_str,"*"*30)
		# print("*"*30,"*"*30)
		if (total_positive > 0) & (total_negative > 0):
			print("Total predictions: ", total_positive+total_negative)
			print("Long - correct:", correct_positive_trend_pred, "total:", total_positive, "percentage:", np.round((correct_positive_trend_pred/total_positive) * 100, 2), "%")
			print("Short - correct:", correct_negative_trend_pred, "total:", total_negative, "percentage:", np.round((correct_negative_trend_pred/total_negative) * 100, 2), "%")
			print("Directional accuracy:", np.round((correct_negative_trend_pred+ correct_positive_trend_pred)/(total_positive+total_negative) * 100,2), "%")
		else:
			print("Long - correct:", correct_positive_trend_pred, "total:", total_positive, )
			print("Short - correct:", correct_negative_trend_pred, "total:", total_negative, )
		# print("*"*65)
		save_model = True
		return save_model

	def create_state(self):
		# print('\n---create state..', self.global_timestep, self.np_train.shape, self.date[self.global_timestep])
		### if continuous
		self.normalised_state = np.float32(self.np_train[self.global_timestep])
		# print(self.normalised_state)

		if self.action_space_str == 'discrete':
			state_discrete = np.where(self.normalised_state >= 0, 1, -1)
			state = np.ravel(state_discrete)
			state_predictions_np = np.array(state)
			# state = np.ravel(np.matmul(state_predictions_np.reshape((len(state),1)), state_predictions_np.reshape((1,len(state)) )))
			state = np.matmul(state_predictions_np.reshape((len(state),1)), state_predictions_np.reshape((1,len(state)) ))
			state = np.ravel(state)
			self.normalised_state = np.float32(state)

		# print(self.normalised_state)

	def np_conversions(self):
		# df_train = df_train.rename_axis('date').reset_index()
		# df_labels = df_labels.rename_axis('date').reset_index()
		# self.train_date_index = df_train.columns.get_loc("date")
		# self.labels_date_index = df_labels.columns.get_loc("date")
		self.labels_target_index = self.obj_dataset.y_train.columns.get_loc("target")
		self.labels_direction_index = self.obj_dataset.y_train.columns.get_loc("direction")
		# self.labels_direction_index = df_labels.columns.get_loc("target")

		assert (self.obj_dataset.X_train.index == self.obj_dataset.y_train.index).all(), "df_train and df_labels have different datetimes"

		self.np_train = self.obj_dataset.X_train.to_numpy()
		self.np_labels = self.obj_dataset.y_train.to_numpy()
		# print(self.np_labels)

		# assert self.np_train[0][self.train_date_index] == self.np_labels[0][self.labels_date_index], 'start date of train and label is different'
		# assert self.np_train[-1][self.train_date_index] == self.np_labels[-1][self.labels_date_index], 'end date of train and label is different'
		assert self.np_train.shape[0] == self.np_labels.shape[0], 'train and labels are of different lengths, please correct them.'

	def load_time_horizon_data(self):
		# print(self.obj_dataset.X_train)
		# print(self.obj_dataset.np_data)

		### convert between start and end date of training data
		start_date = np.datetime64(self.obj_dataset.X_train.index[0]) 
		end_date = np.datetime64(self.obj_dataset.X_train.index[-1]) + np.timedelta64(self.time_horizon_minutes, 'm')

		# Convert the first column to datetime dtype
		self.obj_dataset.np_data[:, 0] = np.array(self.obj_dataset.np_data[:, 0], dtype='datetime64')
		
		# Limit the NumPy array between two dates
		self.np_ohlcv = self.obj_dataset.np_data[(self.obj_dataset.np_data[:, 0] >= start_date) & (self.obj_dataset.np_data[:, 0] <= end_date)]

		# index of np_ohlcv
		self.np_ohlcv_date_index = self.obj_dataset.np_data_date_index 
		self.np_ohlcv_open_index = self.obj_dataset.np_data_open_index 
		self.np_ohlcv_high_index = self.obj_dataset.np_data_high_index 
		self.np_ohlcv_low_index = self.obj_dataset.np_data_low_index 
		self.np_ohlcv_close_index = self.obj_dataset.np_data_close_index 
		self.np_ohlcv_volume_index = self.obj_dataset.np_data_volume_index


		# # limited_array = self.obj_dataset.np_data[(df.index >= start_date) & (df.index <= end_date)]

		# start_date = pd.to_datetime(self.obj_dataset.X_train.index[0]) 
		# end_date = pd.to_datetime(self.obj_dataset.X_train.index[-1]) + pd.DateOffset(hours=self.time_horizon)
		# data = data[ data.date.between(start_date, end_date)]

		# self.P_close     = data['close'].to_numpy()
		# self.P_open = data['open'].to_numpy()
		# self.P_low = data['low'].to_numpy()
		# self.P_high = data['high'].to_numpy()
		# self.date = data['date'].to_numpy()
		# self.P_volume_currency = data['volume'].to_numpy()

		# self.date = [date_obj.strftime('%Y-%m-%d %H:%M:%S+00:00') for date_obj in self.date]

def main():
	print()

if __name__ == "__main__":
	main()
