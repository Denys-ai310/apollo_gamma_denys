import warnings
warnings.filterwarnings("ignore")

from dydx3 import Client
from dydx3.constants import MARKET_BTC_USD, MARKET_ETH_USD, MARKET_TRX_USD, MARKET_SUSHI_USD, \
							MARKET_RUNE_USD, MARKET_CELO_USD, MARKET_XTZ_USD, MARKET_AVAX_USD, \
							MARKET_ZRX_USD, MARKET_EOS_USD, MARKET_ENJ_USD, MARKET_DOT_USD, \
							MARKET_DOGE_USD, MARKET_COMP_USD, MARKET_ZEC_USD, MARKET_ALGO_USD, \
							MARKET_ADA_USD, MARKET_FIL_USD, MARKET_ICP_USD, MARKET_NEAR_USD, \
							MARKET_SNX_USD, MARKET_UMA_USD, MARKET_XLM_USD, MARKET_ONEINCH_USD, \
							MARKET_LINK_USD, MARKET_YFI_USD, MARKET_SOL_USD, MARKET_MATIC_USD, MARKET_LTC_USD, MARKET_MKR_USD 
import datetime
import pandas as pd
import os
import numpy as np
import time
import json

# Neurog wallet's API Keys
KEY = "c789c581-118a-dcab-30ea-f68a12c8c96d"
SECRET = "l3mSl622kz1olKKwKQVKMW3PyVoZlgWP-dChMMYS"
PASSPHRASE = "2nc1xxuD_DImEfaOO5oa"
PUBLIC_KEY_COORDINATES = "060926e7ff105eab366785400416831e4ee4a9a889a51cac62ada7f278451efd"
PUBLIC_KEY = "03cff0fcab74a25f127ef8e63a2a69743787f368d9e1dd799897b474a31e36a2"
STARK_PRIVATE_KEY = "074caeea15b211f705eacf5e3734490760663ad831a844f653a0b728e17d60d0"
ETHEREUM_ADDRESS = "0xF47D78D03311A72c5327678BB00Fd80a1931257c"


class FetchDyDxData:
	def __init__(self, tickers, years_of_data):
		self.tickers = tickers
		self.years_of_data = years_of_data
		curr_dir = os.path.dirname(os.path.realpath(__file__))
		# data_api_keys = open(os.path.join(curr_dir, "data_api_keys.json"))
		# api_keys = json.load(data_api_keys)

		self.dydx_client = Client(
						host='https://api.dydx.exchange',
						api_key_credentials={"key": KEY,
											"secret": SECRET,
											"passphrase": PASSPHRASE,
					},
						stark_private_key=STARK_PRIVATE_KEY,
						default_ethereum_address=ETHEREUM_ADDRESS,
					)
		# self.dydx_client = Client(
		# 				host='https://api.dydx.exchange',
		# 				api_key_credentials={"key": api_keys["dydx"]["key"],
		# 									"secret": api_keys["dydx"]["secret"],
		# 									"passphrase": api_keys["dydx"]["passphrase"],
		# 			},
		# 				stark_private_key=api_keys["dydx"]["stark_private_key"],
		# 				default_ethereum_address=api_keys["dydx"]["ethereum_address"]
		# 			)
		
		curr_dir = os.path.dirname(os.path.realpath(__file__))
		self.exchange_dir = os.path.join(curr_dir, "dydx")
		if not os.path.exists(self.exchange_dir):
			os.makedirs(self.exchange_dir)
		### create tickers directories
		for ticker in tickers:
			ticker = str(ticker).split('-')[0].lower()
			ticker_dir = os.path.join(self.exchange_dir, ticker)
			if not os.path.exists(ticker_dir):
				os.makedirs(ticker_dir)

		# self.data_dir = os.path.join(curr_dir, f"dydx")
		


	def clean_and_ffill(self, df):
		start_date = df.index[0]
		end_date = df.index[-1]
		idx = pd.date_range(start_date, end_date, freq = "1min")
		df = df.reindex(idx, fill_value = np.nan)
		df.ffill(axis = 0, inplace = True)
		df = df[~df.index.duplicated(keep='first')]

		return df

	def run(self):
		for cur_index in range(len(self.tickers)):
			df = pd.DataFrame()
			time_interval = 1  # minutes
			max_candles_per_call = 100
			end_date = datetime.datetime.utcnow()

			# print((self.years_of_data * 365 * 24 * 60))
			calls_required = int((((self.years_of_data * 365 * 24 * 60) / time_interval) / max_candles_per_call)) + 1
			print(self.tickers[cur_index], "| Calls Required: ", calls_required)
			while True:
				exception = False
				for i in range(calls_required):
					if i % 100 == 0 and i != 0:
						print("Calls completed", i)

					start_date = end_date - datetime.timedelta(minutes=time_interval * max_candles_per_call)

					start_date_ = start_date.isoformat()
					end_date_ = end_date.isoformat()

					try:
						candles = self.dydx_client.public.get_candles(
									market=self.tickers[cur_index],
									resolution='1MIN',
									from_iso=start_date_,
									to_iso=end_date_,
									limit=max_candles_per_call
									)
					except:
						print("EXCEPTION")
						time.sleep(10)
						exception = True
						break

					temp = pd.DataFrame.from_records(candles.data['candles'])
					df = df.append(temp)
					# print(df)

					end_date = start_date
				
				if not exception:
					break

			df = df[:-1]
			df.reset_index(inplace=True, drop=True)
			df = df[::-1]
			# print(df)
			df.index = pd.to_datetime(df["startedAt"])
			df = self.clean_and_ffill(df)
			df.reset_index(inplace=True, drop=True)

			df = df[['startedAt', 'open', 'high', 'low', 'close', 'usdVolume']]
			df.rename(columns={'startedAt': 'datetime', 'usdVolume':'volume'}, inplace=True)
			df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
			# print(df)
			
			str_ticker = self.tickers[cur_index].split('-')[0].lower().replace('1', 'one')
			year_started = df.iloc[0]['datetime'].year
			file_name = str_ticker + "_1m_" + str(year_started) + "_onwards.csv"
			df.to_csv(os.path.join(self.exchange_dir, str_ticker, file_name), index=False)


def main():
	tickers = [MARKET_ETH_USD]
	# tickers = [MARKET_BTC_USD, MARKET_ETH_USD, MARKET_TRX_USD, MARKET_SUSHI_USD, MARKET_RUNE_USD, MARKET_CELO_USD, MARKET_XTZ_USD, MARKET_AVAX_USD, MARKET_ZRX_USD, MARKET_EOS_USD, MARKET_ENJ_USD, MARKET_DOT_USD, MARKET_DOGE_USD, MARKET_COMP_USD, MARKET_ZEC_USD, MARKET_ALGO_USD, MARKET_ADA_USD, MARKET_FIL_USD, MARKET_ICP_USD, MARKET_NEAR_USD, MARKET_SNX_USD, MARKET_UMA_USD, MARKET_XLM_USD, MARKET_ONEINCH_USD, MARKET_LINK_USD, MARKET_YFI_USD, MARKET_SOL_USD, MARKET_MATIC_USD, MARKET_LTC_USD, MARKET_MKR_USD]

	years_of_data = 3.6

	obj_fetch_dydx_data = FetchDyDxData(tickers, years_of_data)
	obj_fetch_dydx_data.run()

if __name__=='__main__':
	main()