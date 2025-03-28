import configparser
import warnings
import os
from pathlib import Path

root_dir = Path(__file__).parent.parent.absolute()
# print(root_dir)
	
# def get_data_path(path):
# 	return (root_dir + str(path))

def get_hourly_utc(instrument, exchange, time_horizon, utc):
	# return os.path.join(root_dir, 'data', exchange, instrument, f'btc_{time_horizon}h_2012_onwards_utc{utc}.csv') 
	return os.path.join(root_dir, 'data', exchange, instrument, f'btc_{time_horizon}_2012_onwards_utc{utc}.csv') 

def get_minutes(instrument, exchange):
	if instrument == 'btc':
		return os.path.join(root_dir, 'data', exchange, instrument, 'btc_1m_2012_onwards.csv') 
	else:
		return os.path.join(root_dir, 'data', exchange, instrument, f'{instrument}_1m.csv') 
	

def get_minutes_2021_onwards(instrument, exchange):
	return os.path.join(root_dir, 'data', exchange, instrument, 'btc_1m_2021_onwards.csv') 

def get_minutes_2022_onwards(instrument, exchange):
	return os.path.join(root_dir, 'data', exchange, instrument, 'btc_1m_2022_onwards.csv') 

def get_minutes_2023_onwards(instrument, exchange):
	return os.path.join(root_dir, 'data', exchange, instrument, 'btc_1m_2023_onwards.csv') 
