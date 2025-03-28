import pandas as pd
import math
import numpy as np
import os
import sys
# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils import paths
from scipy import stats
import warnings
import quantstats as qs
warnings.filterwarnings('ignore')

class Backtest():
	def __init__(self, df_predctions, instrument_exchange, take_profit=100, stop_loss=100, buy_after_minutes=0, transaction_fee=0.05, leverage=1):
		self.df_btc_1m = pd.read_csv(paths.get_minutes_2022_onwards(instrument_exchange[0], instrument_exchange[1]))
		self.df_btc_1m = self.df_btc_1m.rename(columns={
				'Date': 'datetime',
				'Low': 'low',
				'High': 'high',
				'Open': 'open',
				'Close': 'close',
				'Volume': 'volume'
			})
		
		self.df_btc_1m['datetime'] = pd.to_datetime(self.df_btc_1m['datetime'])

		self.minute_data_check(df_predctions)

		if len(df_predctions) > 0:
			t1 = pd.to_datetime(df_predctions['datetime'].iloc[0])
			t2 = pd.to_datetime(df_predctions['datetime'].iloc[1])
			self.time_horizon_hours = int(pd.Timedelta(t2 - t1).total_seconds() / 3600)

		self.take_profit_percent = take_profit / 100
		self.stop_loss_percent = stop_loss / 100
		self.buy_after_minutes = int(buy_after_minutes) #in minutes
		self.transaction_fee_percent = transaction_fee 
		self.in_position = False
		self.array_to_save = []						
		self.header_names = [
						'datetime',
						'predicted_direction',
						'action',
						'buy_price',
						'sell_price',
						'balance',
						'pnl'
					]									
 								
	def buy(self, np_temp):
		self.buy_price = np_temp[self.buy_after_minutes][self.min_data_open_index]
		pnl = self.transaction_fee_percent * -1
		buy_transaction_fee = self.current_balance * (pnl/100)
		self.current_balance = self.current_balance + buy_transaction_fee
		self.in_position = True
		self.array_to_save.append([ np_temp[self.buy_after_minutes][self.min_data_date_index], 
									'long' if self.current_pred_direction > 0 else 'short',
									'buy', self.buy_price, 0, self.current_balance,	pnl	] )

	def pnl_direction_change(self, sell_datetime):
		if self.in_position:
			pnl = 0
			if self.previous_pred_direction > 0:
				pnl = ((self.sell_price - self.buy_price)/self.buy_price) * 100
				pnl = pnl - (self.transaction_fee_percent) 
			else:
				pnl = ((self.buy_price - self.sell_price)/self.buy_price) * 100
				pnl = pnl - (self.transaction_fee_percent) 

			self.current_balance += self.current_balance * (pnl/100)
			self.in_position = False
			
			self.array_to_save.append( [sell_datetime, 
										'long' if self.current_pred_direction > 0 else 'short',
										'sell - direction change',
										self.buy_price,
										self.sell_price,
										self.current_balance,
										pnl
										]
									)
		
	def find_tp_sl_index(self, take_profit_amount, stop_loss_amount, np_temp_high, np_temp_low):
		if self.current_pred_direction > 0:
			list_minute_high_indices = np.where(np_temp_high >= take_profit_amount)[0]
			list_minute_low_indices = np.where(np_temp_low <= stop_loss_amount)[0]
		else:
			list_minute_high_indices = np.where(np_temp_high >= stop_loss_amount)[0]
			list_minute_low_indices = np.where(np_temp_low <= take_profit_amount )[0]

		if len(list_minute_high_indices) == 0 and len(list_minute_low_indices) == 0:
			return False, -1
		elif len(list_minute_high_indices) > 0 and len(list_minute_low_indices) == 0:
			df_index = list_minute_high_indices[0]
			self.sell_price = np_temp_high[df_index]
			return True, df_index
		
		elif len(list_minute_high_indices) == 0 and len(list_minute_low_indices) > 0:
			df_index = list_minute_low_indices[0]
			self.sell_price = np_temp_low[df_index]
			return True, df_index
		else:
			if list_minute_high_indices[0] < list_minute_low_indices[0]:
				df_index = list_minute_high_indices[0]
				self.sell_price = np_temp_high[df_index]	
				return True, df_index
			else:
				df_index = list_minute_low_indices[0]
				self.sell_price = np_temp_low[df_index]
				return True, df_index

	def check_tp_sl(self, np_temp, np_temp_high, np_temp_low):
		if self.in_position:
			tp_sl_condition = False
			if self.current_pred_direction > 0: #long
				take_profit_amount = self.buy_price + (self.buy_price * self.take_profit_percent) 
				stop_loss_amount = self.buy_price - (self.buy_price * self.stop_loss_percent)
				tp_sl_condition, df_temp_index = self.find_tp_sl_index(take_profit_amount, stop_loss_amount, np_temp_high, np_temp_low)
			else:
				take_profit_amount = self.buy_price - (self.buy_price * self.take_profit_percent)
				stop_loss_amount = self.buy_price + (self.buy_price * self.stop_loss_percent)
				tp_sl_condition, df_temp_index = self.find_tp_sl_index(take_profit_amount, stop_loss_amount, np_temp_high, np_temp_low)

			if tp_sl_condition:				
				pnl = 0
				if self.previous_pred_direction > 0:
					pnl = ((self.sell_price - self.buy_price)/self.buy_price) * 100
					pnl = pnl - (self.transaction_fee_percent) 
				else:
					pnl = ((self.buy_price - self.sell_price)/self.buy_price) * 100
					pnl = pnl - (self.transaction_fee_percent) 

				self.current_balance += self.current_balance * (pnl/100)
				self.in_position = False

				str_tp_sl = ''
				if pnl > 0:
					str_tp_sl = ' - take_profit'
				else:
					str_tp_sl = ' - stop_loss'

				self.array_to_save.append( [np_temp[df_temp_index][self.min_data_date_index], 
										'long' if self.current_pred_direction > 0 else 'short',
										'sell' + str_tp_sl, 
										self.buy_price,
										self.sell_price,
										self.current_balance,
										pnl
										]
									)


	def run(self):			
		### variable initialisation
		self.pnl_percent_all = 0
		np_model_predctions = self.np_model_predctions
		self.starting_balance = self.current_balance = 1000
		self.breaking_balance = self.current_balance * 0.5
		self.array_to_save = []
		self.in_position = False
		break_on_huge_loss = False

		### first direction
		self.previous_pred_direction = self.current_pred_direction = np_model_predctions[0][self.predicted_direction_index] 
		
		for i in range (0, len(np_model_predctions)-1):
			self.current_pred_direction = np_model_predctions[i][self.predicted_direction_index] 

			if self.current_pred_direction == 0:
				self.current_pred_direction = self.previous_pred_direction
				# continue

			### get minutes data for the current prediction time using numpy
			start_time = np.datetime64(np_model_predctions[i][self.date_index] )
			end_time   = np.datetime64(np_model_predctions[i+1][self.date_index] )
			np_min_data_indices = np.where( (self.np_btc_1m_date >= start_time) & (self.np_btc_1m_date < end_time) )[0]
			np_temp = self.np_btc_1m[np_min_data_indices]

			### numpy minutes data sanity check
			if i == 0:
				np_hour = pd.Timestamp(start_time)
				df_hour = pd.to_datetime(np_model_predctions[i][self.date_index])
				if np_hour.hour != df_hour.hour:
					print('\n Wrong minutes data selected, please numpy conversion of datetime.')
					break

			if self.in_position:
				if self.previous_pred_direction == self.current_pred_direction:
					self.previous_pred_direction = self.current_pred_direction

					self.array_to_save.append( 
								[ np_temp[self.buy_after_minutes][self.min_data_date_index], 
									'long' if self.current_pred_direction > 0 else 'short',
									'same direction',  0, 0, self.current_balance,	0
								]
							)
			### buy
			if not self.in_position: 
				self.buy(np_temp)
				self.previous_pred_direction = self.current_pred_direction
				
			### sell -> change in direction
			if self.current_pred_direction != self.previous_pred_direction: 
				self.sell_price = np_temp[self.buy_after_minutes][self.min_data_open_index]
				sell_datetime = np_temp[self.buy_after_minutes][self.min_data_date_index]

				self.pnl_direction_change(sell_datetime)
				self.previous_pred_direction = self.current_pred_direction

				### buy again after direction change
				if not self.in_position: #buy
					self.buy(np_temp)
					self.previous_pred_direction = self.current_pred_direction


			### get minutes high and low data for the current prediction time using numpy
			np_temp_high = self.np_btc_1m_high[np_min_data_indices]
			np_temp_low = self.np_btc_1m_low[np_min_data_indices]

			### check if during the time horizon it hits take profit or stop loss
			self.check_tp_sl(np_temp, np_temp_high, np_temp_low) 

			self.previous_pred_direction = self.current_pred_direction

			if self.current_balance < self.breaking_balance:
				break_on_huge_loss = True
				break
		
		### backtest dataframe
		df_backtest = pd.DataFrame(self.array_to_save, columns = self.header_names)
		df_backtest["pnl_sum"] = df_backtest["pnl"].cumsum()
		### sharpe
		if len(df_backtest) > 1:
			try:
				sharpe = (qs.stats.sharpe(df_backtest["pnl"], rf=0., periods=365, annualize=True, smart=False))
			except:
				sharpe = -10000
		else:
			sharpe = -10000

		### pnl percent
		pnl_percent = np.round(df_backtest["pnl_sum"].iloc[-1], 2)
		### r2 score
		if len(df_backtest) > 2 :
			res = stats.linregress(range(len(df_backtest.pnl_sum)), df_backtest.pnl_sum.to_numpy())
			r2 = res.rvalue**2
			if pnl_percent < 0:
				r2 = r2*-1
		else:
			r2 = -100000

		if math.isnan(sharpe):
			sharpe = -100000
		if math.isnan(r2):
			r2 = -100000

		


		if break_on_huge_loss:
			return df_backtest, -100000,  round(sharpe, 2),  round(r2, 2),  round(pnl_percent, 2)
		else:
			return df_backtest, round(self.current_balance, 2), round(sharpe, 2), round(r2, 2), round(pnl_percent, 2)

	def minute_data_check(self, df_model_predctions):
		df_model_predctions['datetime'] = df_model_predctions.index
		self.date_index = df_model_predctions.columns.get_loc("datetime")
		self.predicted_direction_index = df_model_predctions.columns.get_loc("predicted_direction")

		df_model_predctions['datetime'] = pd.to_datetime(df_model_predctions['datetime'])
		self.df_model_predctions = df_model_predctions
		self.np_model_predctions = df_model_predctions.to_numpy()
		
		start_date, end_date = df_model_predctions.iloc[0]['datetime'], df_model_predctions.iloc[-1]['datetime']

		start_date = df_model_predctions.iloc[0]['datetime'].tz_localize(None)    	
		end_date = df_model_predctions.iloc[-1]['datetime'].tz_localize(None)

		self.df_btc_1m['datetime'] = pd.to_datetime(self.df_btc_1m['datetime'])

		print("start_date", start_date)
		print("end_date", end_date)

		self.df_btc_1m = self.df_btc_1m[self.df_btc_1m.datetime.between(start_date, end_date)]
		self.df_btc_1m.drop_duplicates(subset = "datetime", keep = 'first', inplace = True)
		self.np_btc_1m = self.df_btc_1m.to_numpy()
		self.np_btc_1m_high = self.df_btc_1m.high.to_numpy()
		self.np_btc_1m_low = self.df_btc_1m.low.to_numpy()
		self.np_btc_1m_date = self.df_btc_1m.datetime.to_numpy().astype(np.datetime64)
		
		# assert start_date in self.df_btc_1m['date'].values, "start date is not in minute data"
		# assert end_date in self.df_btc_1m['date'].values, "end date is not in minute data"
		# assert (self.df_btc_1m.datetime == pd.Timestamp(start_date)).any(), "start date is not in minute data"
		# assert (self.df_btc_1m.datetime == pd.Timestamp(end_date)).any(), f"end date is not in minute data {end_date}"
		
		self.min_data_open_index = self.df_btc_1m.columns.get_loc("open")
		self.min_data_high_index = self.df_btc_1m.columns.get_loc("high")
		self.min_data_low_index = self.df_btc_1m.columns.get_loc("low")
		self.min_data_date_index = self.df_btc_1m.columns.get_loc("datetime")

def main():
	print ("***********************************Backtest*************************************")
	import os
	current_dir = os.path.dirname(os.path.realpath(__file__))
	df_predictions = pd.read_csv(os.path.join(current_dir, 'predicted_direction.csv'))
	print(df_predictions)

	instrument_exchange = ['btc', 'dydx']
	bt = Backtest(df_predictions, instrument_exchange)
	df_backtest, current_balance, sharpe, r2, pnl_percent = bt.run()
	df_backtest.to_csv(os.path.join(current_dir, 'backtest_result.csv'))
	print(df_backtest)

if __name__ == "__main__":
	main()

