import sys
import os
from pathlib import Path
from apollo_gamma.utils import paths

import warnings
warnings.filterwarnings("ignore")

from dydx3.constants import MARKET_BTC_USD, MARKET_ETH_USD, MARKET_TRX_USD, MARKET_SUSHI_USD, \
                            MARKET_RUNE_USD, MARKET_CELO_USD, MARKET_XTZ_USD, MARKET_AVAX_USD, \
                            MARKET_ZRX_USD, MARKET_EOS_USD, MARKET_ENJ_USD, MARKET_DOT_USD, \
                            MARKET_DOGE_USD, MARKET_COMP_USD, MARKET_ZEC_USD, MARKET_ALGO_USD, \
                            MARKET_ADA_USD, MARKET_FIL_USD, MARKET_ICP_USD, MARKET_NEAR_USD, \
                            MARKET_SNX_USD, MARKET_UMA_USD, MARKET_XLM_USD, MARKET_ONEINCH_USD, \
                            MARKET_LINK_USD, MARKET_YFI_USD, MARKET_SOL_USD, MARKET_MATIC_USD, MARKET_LTC_USD
from dydx3 import Client
import pandas as pd
import datetime
import os
import numpy as np
import math
from pathlib import Path

# Neurog wallet's API Keys
KEY = "c789c581-118a-dcab-30ea-f68a12c8c96d"
SECRET = "l3mSl622kz1olKKwKQVKMW3PyVoZlgWP-dChMMYS"
PASSPHRASE = "2nc1xxuD_DImEfaOO5oa"
PUBLIC_KEY_COORDINATES = "060926e7ff105eab366785400416831e4ee4a9a889a51cac62ada7f278451efd"
PUBLIC_KEY = "03cff0fcab74a25f127ef8e63a2a69743787f368d9e1dd799897b474a31e36a2"
STARK_PRIVATE_KEY = "074caeea15b211f705eacf5e3734490760663ad831a844f653a0b728e17d60d0"
ETHEREUM_ADDRESS = "0xF47D78D03311A72c5327678BB00Fd80a1931257c"

class UpdateDydxData:
    def __init__(self, tickers):
        self.tickers = tickers
                                
        self.dydx_client = Client(
                        host='https://api.dydx.exchange',
                        api_key_credentials={"key": KEY,
                                            "secret": SECRET,
                                            "passphrase": PASSPHRASE,
                    },
                        stark_private_key=STARK_PRIVATE_KEY,
                        default_ethereum_address=ETHEREUM_ADDRESS,
                    )

    def clean_and_ffill(self, df, frequency):
        start_date = df.index[0]
        end_date = df.index[-1]

        idx = pd.date_range(start_date, end_date, freq=frequency)

        df = df.reindex(idx, fill_value = np.nan)

        df.ffill(axis = 0, inplace = True)
        df = df[~df.index.duplicated(keep='first')]

        return df

    def update_dydx_data(self, ticker):
        resolution = "1MIN"
        frequency = "1min"
        exchange = "dydx"
        utc = 0
        df = pd.DataFrame()
        str_ticker = ticker.split('-')[0].lower().replace('1', 'one')
        df_data = pd.read_csv(paths.get_minutes(str_ticker, exchange))
        df_data["datetime"] = pd.to_datetime(df_data["datetime"])
        
        while True:
            df_data_start_date = df_data.iloc[-1]["datetime"]
            # df_data_start_date = pd.to_datetime("2023-06-12 08:52:00+00:00")
            start_date = df_data_start_date - datetime.timedelta(minutes=5)  # need to start fetching data from this point

            date_diff = (datetime.datetime.now(datetime.timezone.utc) - start_date)
            timedelta_minutes = ((date_diff.days * 24) * 60) + int(date_diff.seconds / 60)
            calls_required = math.ceil(timedelta_minutes / 100)
            # calls_required = 10

            # print("calls_required", calls_required)
            exception = False
            for i in range(calls_required):
                # if i % 100 == 0 and i != 0:
                    # print(i, df.iloc[-1])

                end_date = start_date + datetime.timedelta(minutes=101)
                
                start_date_ = start_date.isoformat().split('+')[0]
                end_date_ = end_date.isoformat().split('+')[0]

                # print(start_date_, end_date_)
                try:
                    candles = self.dydx_client.public.get_candles(
                                market=ticker,
                                resolution=resolution,
                                from_iso=start_date_,
                                to_iso=end_date_,
                                limit=100
                                )
                except:
                    print("EXCEPTION")
                    exception = True
                    break

                temp = pd.DataFrame.from_records(candles.data['candles'])
                temp = temp[::-1]

                df = df.append(temp)

                start_date = start_date + datetime.timedelta(minutes=100)
            
            if not exception:
                break

        # print(df)

        df["startedAt"] = pd.to_datetime(df["startedAt"])
        df = df[df["startedAt"] > df_data_start_date]    # remove any extra rows

        # forward-filling any missing rows and removing duplicate rows
        df.reset_index(inplace=True, drop=True)
        df.index = pd.to_datetime(df["startedAt"])
        df = self.clean_and_ffill(df, frequency)
        df.reset_index(inplace=True, drop=True)

        df = df[['startedAt', 'open', 'high', 'low', 'close', 'usdVolume']]
        df.rename(columns={'startedAt': 'datetime', 'usdVolume':'volume'}, inplace=True)
        
        ### forward fill if any column contains 0
        for col in df.columns:
            if col != 'datetime':
                df[col] = df[col].astype(float)
                tmp = np.where(df[col].values <= 0)
                if len(tmp[0]) >= 1:
                    if 0 in list(tmp[0]): ## if 0th index is 0 then we need to bfill otherwise, 0 will be used as ffill
                        df[col] = df[col].replace(to_replace=0, method='bfill')
                    else:
                        df[col] = df[col].replace(to_replace=0, method='ffill')
                    print(f'Column {col} contained {len(tmp[0])} zero values which are forward filled')
        
        
        df_data = pd.concat([df_data, df])
        df_data = df_data[:-1]
        minutes_path = paths.get_minutes(str_ticker, exchange)
        df_data.to_csv(minutes_path, index=False)
        print(df_data)

        ## save data from 2022 onwards
        minutes_path_2022 = paths.get_minutes_2022_onwards(str_ticker, exchange)
        start_date = pd.to_datetime('2022-01-01 00:00:00+00:00')
        df_data['datetime'] = pd.to_datetime(df_data['datetime'], utc=True)
        df_2022 = df_data.loc[(df_data.datetime >= start_date)]
        df_2022.to_csv(minutes_path_2022, index=False)

        ### save data for each hour
        intervals = [1, 2, 3, 4, 6, 8, 10, 12, 24]
        for interval in intervals:
            print(f"Creating {str_ticker} {interval} hourly data...")
            df_interval = df_data.groupby(pd.Grouper(key='datetime', freq=f'{interval}H')).agg({"open": "first", 
                                                                                            "high": np.max,
                                                                                            "low": np.min, 
                                                                                            "close": "last",
                                                                                            "volume": np.sum,
                                                                                            }).reset_index()
            hourly_path = paths.get_hourly_utc(str_ticker, exchange, str(interval)+'h', utc)
            df_interval.to_csv(hourly_path, index=False)
            

        ### save data for each mins
        intervals = [10, 30]
        for interval in intervals:
            print(f"Creating {str_ticker} {interval} mins data...")
            df_interval = df_data.groupby(pd.Grouper(key='datetime', freq=f'{interval}Min')).agg({"open": "first", 
                                                                                            "high": np.max,
                                                                                            "low": np.min, 
                                                                                            "close": "last",
                                                                                            "volume": np.sum,
                                                                                            }).reset_index()
            hourly_path = paths.get_hourly_utc(str_ticker, exchange, str(interval)+'min', utc)
            start_date = pd.to_datetime('2012-01-03 00:00:00+00:00')
            df_interval = df_interval.loc[(df_interval.datetime >= start_date)]
            df_interval.to_csv(hourly_path, index=False)

        
            



    def run(self):
        for i in range(len(self.tickers)):
            try:
                print(f"Updating minute data for {self.tickers[i].split('-')[0].replace('1', 'ONE')}")
                self.update_dydx_data(self.tickers[i])
            except:
                continue

def main():
    tickers = [MARKET_BTC_USD, MARKET_ETH_USD,
            #     MARKET_TRX_USD, MARKET_SUSHI_USD, \
            #     MARKET_RUNE_USD, MARKET_CELO_USD, MARKET_XTZ_USD, MARKET_AVAX_USD, \
            #     MARKET_ZRX_USD, MARKET_EOS_USD, MARKET_ENJ_USD, MARKET_DOT_USD, \
            #     MARKET_DOGE_USD, MARKET_COMP_USD, MARKET_ZEC_USD, MARKET_ALGO_USD, \
            #     MARKET_ADA_USD, MARKET_FIL_USD, MARKET_ICP_USD, MARKET_NEAR_USD, \
            #     MARKET_SNX_USD, MARKET_UMA_USD, MARKET_XLM_USD, MARKET_ONEINCH_USD, \
            #     MARKET_LINK_USD, MARKET_YFI_USD, MARKET_SOL_USD, MARKET_MATIC_USD, MARKET_LTC_USD
                ]

    obj_update_data = UpdateDydxData(tickers)
    obj_update_data.run()

if __name__=='__main__':
    main()


