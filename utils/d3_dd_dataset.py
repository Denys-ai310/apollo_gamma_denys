import numpy as np
import os
import shutil
from d3rlpy.dataset import MDPDataset
import pickle
import random
import glob
import torch
from apollo_gamma.utils import utils
config = utils.get_config_file()
nb_samples = 1000

def create(env, dataset_path):
	observations = []
	actions = []
	rewards = []
	terminals = []
	for i in range (nb_samples):
		env.reset_global_timestep()
		state = env.reset()
		observations.append(state)
		while True:
			current_index = env.global_timestep-1
			next_index = env.global_timestep
			action = [env.np_labels[current_index][env.labels_target_index]]
			state, reward, done, aa = env.step(action)
			actions.append(action)
			rewards.append(reward)
			terminals.append(done)
			if done:
				break
			else:
				observations.append(state)
	observations = np.array(observations)
	actions       = np.array(actions)
	rewards = np.array(rewards)
	terminals = np.array(terminals)
	dataset = MDPDataset(observations, actions, rewards, terminals)
	if len(dataset.episodes) > 0:
		# Access the first episode's data directly
		episode = dataset.episodes[0]
		print("First transition data:")
		print("Observation:", observations[0])
		print("Action:", actions[0])
		print("Reward:", rewards[0])
		print("Next observation:", observations[1] if len(observations) > 1 else None)
		print("Terminal:", terminals[0])
	dataset.dump(dataset_path)

def create_ds_optimise(env, dataset_path):
	observations = []
	actions = []
	rewards = []
	terminals = []
	for i in range (int(nb_samples)):
		state = env.reset()
		observations.append(state)
		while True:
			current_index = env.global_timestep-1
			next_index = env.global_timestep
			action = [(env.P_close[next_index] - env.P_close[current_index]) / env.P_close[current_index]]
			state, reward, done, aa = env.step(action)
			actions.append(action)
			rewards.append(reward)
			terminals.append(done)
			if done:
				break
			else:
				observations.append(state)
	observations = np.array(observations)
	actions       = np.array(actions)
	rewards = np.array(rewards)
	terminals = np.array(terminals)
	dataset = MDPDataset(observations, actions, rewards, terminals, discrete_action=False)
	if len(dataset.episodes) > 0:
		first_transition = next(iter(dataset.episodes[0]))
		print("First transition data:")
		print("Observation:", first_transition.observation)
		print("Action:", first_transition.action)
		print("Reward:", first_transition.reward)
		print("Next observation:", first_transition.next_observation)
		print("Terminal:", first_transition.terminal)
	dataset.dump(dataset_path)

def create_ds_discrete(env, dataset_path):
	observations = []
	actions = []
	rewards = []
	terminals = []
	for i in range (nb_samples):
		state = env.reset()
		observations.append(state)
		while True:
			current_index = env.global_timestep-1
			next_index = env.global_timestep
			# action = [(env.P_close[next_index] - env.P_close[current_index]) / env.P_close[current_index]]
			action = 1 if env.P_close[env.global_timestep-1] - env.P_close[env.global_timestep] < 0 else 0
			# if 
			# print(action)
			state, reward, done, aa = env.step(action)
			actions.append(action)
			rewards.append(reward)
			terminals.append(done)
			if done:
				break
			else:
				observations.append(state)
	observations = np.array(observations)
	actions       = np.array(actions)
	rewards = np.array(rewards)
	terminals = np.array(terminals)
	dataset = MDPDataset(observations, actions, rewards, terminals, discrete_action=True)
	if len(dataset.episodes) > 0:
		print("First transition data:")
		print("Observation:", observations[0])
		print("Action:", actions[0])
		print("Reward:", rewards[0])
		print("Next observation:", observations[1] if len(observations) > 1 else None)
		print("Terminal:", terminals[0])
	dataset.dump(dataset_path)

def create_ds_zt(env, dataset_path):
	observations = []
	actions = []
	rewards = []
	terminals = []
	for i in range (nb_samples):
		state = env.reset()
		observations.append(state)
		while True:
			current_index = env.global_timestep-1
			next_index = env.global_timestep
			action = [env.actual_alpha_log_diff]
			state, reward, done, aa = env.step(action)
			# action = [env.rescale_actions(action[0], env.action_range , env.alpha_log_diff_lb_ub)]
			# state, reward, done, aa = env.step([env.actual_alpha_log_diff])
			actions.append(action)
			rewards.append(reward)
			terminals.append(done)
			if done:
				break
			else:
				observations.append(state)
	observations = np.array(observations)
	actions       = np.array(actions)
	rewards = np.array(rewards)
	terminals = np.array(terminals)
	dataset = MDPDataset(observations, actions, rewards, terminals, discrete_action=False)
	dataset.episodes
	episode = dataset.episodes[0]
	episode[0].observation
	episode[0].action
	# episode[0].next_reward
	episode[0].reward
	episode[0].next_observation
	episode[0].terminal
	transition = episode[0]
	while transition.next_transition:
		transition = transition.next_transition
	# save as HDF5
	dataset.dump(dataset_path)

