import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import os
import sys
from apollo_gamma.backtest import backtest
from apollo_gamma.utils import utils
import sqlite3
import os
import shutil
import glob
import sqlite3
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataset import Dataset, random_split
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from stable_baselines3 import PPO, DDPG, TD3, A2C, SAC
import gym
# import timeit
# import optuna
import quantstats as qs
import torch
from sklearn.preprocessing import MinMaxScaler
import warnings
import matplotlib



config = utils.get_config_file()
# models_directory = str(config['training']['models_directory'])
models_directory = 'models'

gamma = 0.9
learning_rate = 0.0005
seed = 818
total_timesteps = 2000
# total_timesteps = 2000
save_frequency = 1000

nb_epochs = 2
# nb_epochs = 5



class CustomSB():
    def __init__(self, arg_vector):
        self.algo = arg_vector[0]
        self.current_dir = arg_vector[1]
        self.train_env = arg_vector[2]
        # self.backtest_env = arg_vector[3]
        # self.realworld_env = arg_vector[4]
        # self.execute_rw_backtest = arg_vector[5]
        self.instrument_exchange_timehorizon_utc = arg_vector[3]
        self.library = arg_vector[4]
        self.obj_dataset = arg_vector[5]
        self.rolling_window_length = arg_vector[6]
        self.transaction_fee = arg_vector[7]
        self.take_profit_percent = arg_vector[8]
        self.stop_loss_percent = arg_vector[9]
        self.leverage = arg_vector[10]
        self.use_gpu = arg_vector[11]

        # arg_vector = [sb_mf_algo_array[i], current_dir, train_env, instrument_exchange_timehorizon_utc, library, obj_dataset]

        ### directory paths for backtest and realworld (results and db)
        self.algo_name = self.library + '_' + str(self.algo).split('.')[-2]

        ### delete model's exisiting data, i.e., results, db, and model predictions
        self.path_bt_results, self.path_bt_db, self.path_ft_results, \
                self.path_ft_db, self.path_for_saving_models  = utils.get_paths_drl(self.current_dir, self.algo_name)

        self.models_directory = os.path.join(self.current_dir, models_directory)

        th = f'{self.instrument_exchange_timehorizon_utc[2]}'
        instrument = f'{self.instrument_exchange_timehorizon_utc[0]}'.upper()
        utc = f'{self.instrument_exchange_timehorizon_utc[3]}'
        ds = f'{self.obj_dataset.train_test_details}'
        algo = f'{self.algo_name}'
        sw = f'{self.obj_dataset.rolling_window_size}'
        # self.model_name_prefix = f'{instrument}_{th}_{algo}_{sw}'
        # print(self.model_name_prefix)
        self.model_name_prefix = f'{instrument}_{th}_{algo}_{sw}_{utc}_{ds}'
        # self.model_name_prefix = f'{th}_{instrument}_{utc}_{ds}_{algo}_{sw}'
        # print(self.model_name_prefix)
        self.table_name = f'{instrument}_{th}'
        # print(self.model_name_prefix)       
        self.best_model = False

        ### delete existing data if exists
        utils.delete_exisiting_data(self.algo_name, self.table_name, self.path_bt_results, self.path_bt_db, self.path_ft_results, self.path_ft_db, self.models_directory)
                
    def train(self):
        #####################################################################################################
        ### training
        #####################################################################################################
        
        if os.path.exists(self.path_for_saving_models) and os.path.isdir(self.path_for_saving_models):
            shutil.rmtree(self.path_for_saving_models)

        device = 'cpu'
        if self.use_gpu :
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.library == 'sb3':
            from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
            model = self.algo("MlpPolicy", self.train_env, verbose=0, gamma=gamma, device=device)

        checkpoint_callback = CheckpointCallback(save_freq=save_frequency, save_path=self.path_for_saving_models, name_prefix=self.model_name_prefix)
        model.learn(total_timesteps=total_timesteps,callback=checkpoint_callback)


        #####################################################################################################
        ### Backtesting
        #####################################################################################################
        # print('backtesting...')
        # bt_records = ft_records = []
        # bt_ledger = ft_ledger = 
        best_model_name = best_model_path = df_best_ledger = None
        tmp_bt_metric = tmp_bt_metric_last = -10000
        list_of_models_path = glob.glob(self.path_for_saving_models +'/*.zip')


        df_input = pd.concat([self.obj_dataset.X_backtest, self.obj_dataset.X_forwardtest])
        
        df_backtest = pd.concat([self.obj_dataset.df_backtest, self.obj_dataset.df_forwardtest])

        # print(df_input)
        # print(df_backtest)
        test_obs = self.train_env.reset()
        # Add preprocessing to match training data
        print("Before preprocessing:", df_input.shape)

        # Select only the first 13 columns if we have more
        # if df_input.shape[1] > test_obs.shape[0]:
        #     df_input = df_input.iloc[:, :test_obs.shape[0]]

        print("After preprocessing:", df_input.shape)
        print("Column names:", df_input.columns.tolist())

        


        ### loop over all saved models during training
        for i in range(len(list_of_models_path)):
            

            tmp_model = self.algo.load(list_of_models_path[i])
            model_name = str(os.path.basename(list_of_models_path[i])).split('.zip')[0]
            # model_name_shortened = model_name.split('_steps')[0]

            # print("df_input:", df_input.shape)
            # if df_input.shape[1] > test_obs.shape[0]:
            #     df_input = df_input.iloc[:, :test_obs.shape[0]]


            ### backtest
            bt_ledger_, bt_ending_balance, bt_sharpe, bt_r2, bt_pnl_percent, df_pred = utils.predict_and_backtest_drl(self.library, \
                                            'backtest', tmp_model, df_input, df_backtest, self.instrument_exchange_timehorizon_utc, self.transaction_fee, self.take_profit_percent, self.stop_loss_percent, self.leverage)
            
            
            

            pred_long_percent, pred_short_percent, acc_overal, acc_long, acc_short = utils.get_accuracies(df_pred, bt_ledger_, self.transaction_fee)

            print('accuracies:', pred_long_percent, pred_short_percent, acc_overal, acc_long, acc_short)
            print('bt_sharp, bt_r2:', bt_sharpe, bt_r2)



            if pred_long_percent > 0.4 and pred_short_percent > 0.4:
                if acc_long > 0.5 and acc_short > 0.5:
                    if bt_sharpe >= 2 and bt_r2 >= 0.85:
                        if bt_sharpe > tmp_bt_metric:
                            # print('rrrr--------------------')
                            # print(pred_long_percent, pred_short_percent, acc_overal, acc_long, acc_short)
                            tmp_bt_metric = bt_sharpe
                            best_model_name = model_name
                            best_model_path = list_of_models_path[i]
                            self.best_model = True
                            df_best_ledger = bt_ledger_
            
            # print(not(self.best_model), self.obj_dataset.rolling_window_size, self.rolling_window_length, bt_sharpe, tmp_bt_metric_last)
            if not(self.best_model):
                if self.obj_dataset.rolling_window_size == self.rolling_window_length:
                    if bt_sharpe > tmp_bt_metric_last:
                        # print(' ************************ ')
                        # print(pred_long_percent, pred_short_percent, acc_overal, acc_long, acc_short)
                        tmp_bt_metric_last = bt_sharpe
                        best_model_name = model_name
                        best_model_path = list_of_models_path[i]
                        df_best_ledger = bt_ledger_
            
            
           


        if not(best_model_name == None):
            

            tmp_model = self.algo.load(best_model_path)
            ### backtest
            bt_ledger, bt_ending_balance, bt_sharpe, bt_r2, bt_pnl_percent, _ = utils.predict_and_backtest_drl(self.library, \
                                            'backtest', tmp_model, self.obj_dataset.X_backtest, self.obj_dataset.df_backtest, self.instrument_exchange_timehorizon_utc, self.transaction_fee, self.take_profit_percent, self.stop_loss_percent, self.leverage)

            bt_record = [self.model_name_prefix, bt_ending_balance, bt_pnl_percent, bt_r2, bt_sharpe, utils.get_ledger_days(bt_ledger)]
            # print(bt_record)

            ### forwardtest
            ft_ledger, ft_ending_balance, ft_sharpe, ft_r2, ft_pnl_percent, _ = utils.predict_and_backtest_drl(self.library, \
                                            'forwardtest', tmp_model, self.obj_dataset.X_forwardtest, self.obj_dataset.df_forwardtest, self.instrument_exchange_timehorizon_utc, self.transaction_fee, self.take_profit_percent, self.stop_loss_percent, self.leverage)
            ft_record = [self.model_name_prefix, ft_ending_balance, ft_pnl_percent, ft_r2, ft_sharpe, utils.get_ledger_days(ft_ledger)]
            # print(ft_record)

            ### copy the model to the main directory
            model_to_copy = os.path.join(self.path_for_saving_models, f'{best_model_name}.zip')
            shutil.copy(model_to_copy, self.models_directory)

            model_to_rename = os.path.join(self.models_directory, f'{best_model_name}.zip')
            best_model_name = best_model_name.split('_')[:-2] 
            best_model_name = "_".join(best_model_name)
            new_path = f"{self.models_directory}/{best_model_name}.zip"        
            shutil.move(model_to_rename, new_path)

            utils.insert_db_drl(self.path_bt_db, self.table_name, [bt_record])
            utils.insert_db_drl(self.path_ft_db, self.table_name, [ft_record])

            bt_ledger.to_csv(os.path.join(self.path_bt_results, f'bt_ledger_{best_model_name}.csv'))
            ft_ledger.to_csv(os.path.join(self.path_ft_results, f'ft_ledger_{best_model_name}.csv'))
            

            # df_best_ledger.set_index('datetime', inplace=True)
            df_best_ledger = df_best_ledger[df_best_ledger['sell_price'] != 0]
            
            df_best_ledger['pnl'] = df_best_ledger['pnl'] + self.transaction_fee
            # df_best_ledger['pnl'] = df_best_ledger['pnl'].astype(float)/100    
            df_best_ledger = df_best_ledger['pnl']/100
            df_best_ledger = df_best_ledger.resample('D').sum()
            print("----------------------------------")
            print("bt_ledger_shape:", df_best_ledger.shape)
            print("bt_ledger_columns:", df_best_ledger.head())
            
            # Convert timezone-aware index to timezone-naive
            df_best_ledger.index = df_best_ledger.index.tz_localize(None)

            
            # df_best_ledger.index = df_best_ledger.index.tz_convert(None)
            path_report = os.path.join(self.models_directory, f'{best_model_name}.html')
            title_report = str(self.algo).split('.')[-2].upper() + ' Results'
            qs.reports.html(df_best_ledger, title=title_report, output=True, compounded=False, download_filename=path_report)

        utils.delete_dir(self.path_for_saving_models)

class GenerateBacktestDB():
    def __init__(self, arg_vector):
        self.algo = arg_vector[0]
        self.dir_for_saving_csvs = arg_vector[1]
        self.algo_name = arg_vector[2]
        self.current_dir = arg_vector[3]
        self.path_db = arg_vector[4]
        self.path_results = arg_vector[5]
        self.list_of_models_path = arg_vector[6]
        self.bt_rw = arg_vector[7]
        self.instrument_exchange_timehorizon_utc = arg_vector[8]
    
    #####################################################################################################
    ### Only those results will be inserted into db which are greator than r2 threshold
    #####################################################################################################
    def meeting_r2_threshold(self):
        ### create db
        # db_path = os.path.join(self.current_dir, self.db_directory)
        # conn = sqlite3.connect(os.path.join(db_path, f'{self.algo_name}_{self.bt_rw}.db'))
        conn = sqlite3.connect(self.path_db)
        c = conn.cursor()
        c.execute(f"DROP TABLE IF EXISTS {self.algo_name}")
        c.execute(f""" CREATE TABLE IF NOT EXISTS 
        \"{self.algo_name}\"(
            model_number DATATYPE,
            balance DATATYPE,
            pnl_percent DATATYPE,
            r2 DATATYPE,
            sharpe DATATYPE
        )
        """)
        conn.commit()

        model_selection_r2_score = 0.9
        ### perform bakctest
        list_of_models_preditions = glob.glob(self.dir_for_saving_csvs +'/*.csv')
        for i in range(len(list_of_models_preditions) ):
            # start = timeit.default_timer()
            df_tmp = pd.read_csv(list_of_models_preditions[i])
            model_predictions_name = list_of_models_preditions[i].split('model_')[-1].split('.')[0]
            obj_backtest = backtest.Backtest(df_tmp, self.instrument_exchange_timehorizon_utc)
            ledger, ending_balance, sharpe, r2, pnl_percent = obj_backtest.run()     
            record = [model_predictions_name, ending_balance, pnl_percent, r2, sharpe]  
            # print(timeit.default_timer()-start)
            
            # print(record)
            ### save ledger and insert into db if r2 is greator than threshold
            if r2 >= model_selection_r2_score and pnl_percent > 0:
                print(record)
                filename = f'{self.algo_name}_{model_predictions_name}_{r2}'
                ledger.to_csv(os.path.join(self.path_results, f'{filename}.csv'))

                ### insert backtest results in the db
                c.execute(f"""INSERT INTO \"{self.algo_name}\"
                                VALUES(?,?,?,?,?);""",record)
                conn.commit()

        ### delete models and csvs with under threshold r2
        run_rw = self.delete_models_csvs(conn)
        conn.close()
        return run_rw

    #####################################################################################################
    ### delete models and their prediction csvs 
    #####################################################################################################
    def delete_models_csvs(self, db_conn):
        ### models that have r2 greator than threshold
        tmp_df = pd.read_sql_query("SELECT * from " + self.algo_name, db_conn)
        run_rw = True
        if tmp_df.empty:
            run_rw = False
        models_to_keep = tmp_df['model_number'].to_numpy()
        list_of_csvs = glob.glob(self.dir_for_saving_csvs +'/*.csv')

        for i in range(len(self.list_of_models_path)):
            model = self.list_of_models_path[i]
            if 'd3rlpy' in self.algo_name:
                model = int(model.split('model_')[-1].split('.')[0])
            else:
                model = int(model.split('model_')[-1].split('_')[0])
            if model not in models_to_keep:
                os.remove(self.list_of_models_path[i])
                os.remove(list_of_csvs[i])

        return run_rw

    #####################################################################################################
    ### run backtest on all csvs in a folder
    #####################################################################################################
    def run(self):
        ### perform bakctest
        list_of_models_preditions = glob.glob(self.dir_for_saving_csvs +'/*.csv')
        records = []
        for i in range(len(list_of_models_preditions) ):
            df_tmp = pd.read_csv(list_of_models_preditions[i])
            model_predictions_name = list_of_models_preditions[i].split('model_')[-1].split('.')[0]
            obj_backtest = backtest.Backtest(df_tmp, self.instrument_exchange_timehorizon_utc)
            ledger, ending_balance, sharpe, r2, pnl_percent = obj_backtest.run()     
            record = [model_predictions_name, ending_balance, pnl_percent, r2, sharpe]  
            records.append(record)

            print(record)

            # if r2 >= model_selection_r2_score:
            #     filename = f'{self.algo_name}_{model_predictions_name}_{r2}'
            #     ledger.to_csv(os.path.join(self.path_results, f'{filename}.csv'))
            filename = f'{self.algo_name}_{model_predictions_name}_{r2}'
            ledger.to_csv(os.path.join(self.path_results, f'{filename}.csv'))

        ### insert backtest results in the db
        # db_path = os.path.join(self.current_dir, self.db_directory)
        # conn = sqlite3.connect(os.path.join(db_path, f'{self.algo_name}.db'))
        conn = sqlite3.connect(self.path_db)
        c = conn.cursor()
        c.execute(f"DROP TABLE IF EXISTS {self.algo_name}")
        c.execute(f""" CREATE TABLE IF NOT EXISTS 
        \"{self.algo_name}\"(
            model_number DATATYPE,
            balance DATATYPE,
            pnl_percent DATATYPE,
            r2 DATATYPE,
            sharpe DATATYPE
        )
        """)
        conn.commit()
        c.executemany(f"""INSERT INTO \"{self.algo_name}\"
                        VALUES(?,?,?,?,?);""",records)
        conn.commit()
        conn.close()

class GenerateBacktestCSVs():
    def __init__(self, arg_vector):
        self.algo = arg_vector[0]
        self.list_of_models_path = arg_vector[1]
        self.env = arg_vector[2]
        self.dir_for_saving_csvs = arg_vector[3]
        self.algo_name = arg_vector[4]
        self.header_names = ['date', 'actual_open', 'actual_close', 'actual_direction', 'predicted_direction', 'predicted_close']

        utils.create_override_dir(self.dir_for_saving_csvs)

    def sb(self):
        # print('Generating csvs for ', self.algo)
        for i in range(len(self.list_of_models_path) ):
            model_to_load = self.list_of_models_path[i]
            # print (model_to_load)
            model = self.algo.load(model_to_load)
            predictions_path = self.dir_for_saving_csvs + '/' + self.algo_name +  '_model_' + model_to_load.split('model_')[-1].split('_')[0] + '.csv'

            self.env.reset_global_timestep()
            obs = self.env.reset()
            array_to_save = []
            save_model = False
            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, return_array, save_model = self.env.step(action)
                array_to_save.append(return_array)
                if dones:
                    break
            del model 
            if save_model:
                df = pd.DataFrame(array_to_save, columns = self.header_names) 
                df.to_csv(predictions_path)

    def d3(self, dir_for_saving_models):
        import numpy as np
        import torch
        import d3rlpy
        import random
        # set random seeds in random module, numpy module and PyTorch module.
        # seed = int(config['training']['seed'])
        # d3rlpy.seed(seed)
        # torch.manual_seed(seed)
        # np.random.seed(seed)
        # random.seed(seed)

        # print('Generating csvs for ', self.algo)
        for i in range(len(self.list_of_models_path) ):
            model_to_load = self.list_of_models_path[i]
            model = self.algo.from_json(dir_for_saving_models +'/params.json')
            model.load_model(model_to_load) 
            predictions_path = self.dir_for_saving_csvs + '/' + self.algo_name +  '_model_' + model_to_load.split('model_')[-1].split('.')[0] + '.csv'

            self.env.reset_global_timestep()
            obs = self.env.reset()
            obs = np.array([obs], dtype= np.float32)
            obs = torch.from_numpy(obs)
            array_to_save = []
            save_model = False
            while True:
                action = model.predict(obs)[0]
                obs, rewards, dones, return_array, save_model = self.env.step(action)
                obs = np.array([obs], dtype= np.float32)
                obs = torch.from_numpy(obs)
                array_to_save.append(return_array)
                if dones:
                    break
            del model 
            if save_model:
                df = pd.DataFrame(array_to_save, columns = self.header_names) 
                df.to_csv(predictions_path)


# arg_vector = [d3_algo, current_dir, train_env, instrument_exchange_timehorizon_utc, library, dataset_path, obj_dataset, algo_name, rolling_window_length, ec.transaction_fee]

class CustomD3RLPY():    
    def __init__(self, arg_vector):
        self.algo = arg_vector[0]
        self.current_dir = arg_vector[1]
        self.train_env = arg_vector[2]
        # self.backtest_env = arg_vector[3]
        # self.realworld_env = arg_vector[4]
        # # self.execute_rw_backtest = arg_vector[5]
        self.instrument_exchange_timehorizon_utc = arg_vector[3]
        self.library = arg_vector[4]
        self.dataset_path = arg_vector[5]
        # self.bt_env_d3 = arg_vector[8]
        self.obj_dataset = arg_vector[6]
        self.rolling_window_length = arg_vector[7]
        self.transaction_fee = arg_vector[8]
        self.take_profit_percent = arg_vector[9]
        self.stop_loss_percent = arg_vector[10]
        self.leverage = arg_vector[11]
        self.use_gpu = arg_vector[12]

        ### directory paths for backtest and realworld (results and db)
        self.algo_name = self.library + '_' + str(self.algo).split('.')[-2]

        ### delete model's exisiting data, i.e., results, db, and model predictions
        self.path_bt_results, self.path_bt_db, self.path_ft_results, \
                self.path_ft_db, self.path_for_saving_models  = utils.get_paths_drl(self.current_dir, self.algo_name)

        self.models_directory = os.path.join(self.current_dir, models_directory)

        ### name prefix for models saving
        th = f'{self.instrument_exchange_timehorizon_utc[2]}'
        instrument = f'{self.instrument_exchange_timehorizon_utc[0]}'.upper()
        utc = f'{self.instrument_exchange_timehorizon_utc[3]}'
        ds = f'{self.obj_dataset.train_test_details}'
        algo = f'{self.algo_name}'
        sw = f'{self.obj_dataset.rolling_window_size}'
        # self.model_name_prefix = f'{th}_{instrument}_{utc}_{ds}_{algo}_{sw}'
        # print(self.model_name_prefix)
        # self.model_name_prefix = f'{instrument}_{th}_{algo}_{sw}'
        
        self.model_name_prefix = f'{instrument}_{th}_{algo}_{sw}_{utc}_{ds}'

        self.table_name = f'{instrument}_{th}'
        # print(self.model_name_prefix)      
        # print('td3_plus_bc' in self.model_name_prefix)      
        if 'td3_plus_bc' in self.model_name_prefix:
            self.model_name_prefix = self.model_name_prefix.replace('td3_plus_bc', 'td3plusbc')
        # print(self.model_name_prefix)      
        self.best_model = False

        ### delete existing data if exists
        utils.delete_exisiting_data(self.algo_name, self.table_name, self.path_bt_results, self.path_bt_db, self.path_ft_results, self.path_ft_db, self.models_directory)

    def train(self):

        # import logging
        # import warnings
        # logging.getLogger('lightning').setLevel(0)
        # logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
        # warnings.simplefilter(action='ignore', category=FutureWarning)
        # warnings.filterwarnings('ignore')
        # import warnings
        # from torch.serialization import SourceChangeWarning
        # warnings.filterwarnings("ignore", category=SourceChangeWarning)

        #####################################################################################################
        ### training
        #####################################################################################################
        # print('training...')
        if os.path.exists(self.path_for_saving_models) and os.path.isdir(self.path_for_saving_models):
            shutil.rmtree(self.path_for_saving_models)

        from d3rlpy.dataset import MDPDataset
        from sklearn.model_selection import train_test_split
        from d3rlpy.metrics.scorer import evaluate_on_environment, td_error_scorer, value_estimation_std_scorer, continuous_action_diff_scorer, discounted_sum_of_advantage_scorer, average_value_estimation_scorer, discrete_action_match_scorer
        # nb_epochs = int(config['drl_agents_parameters']['nb_epochs'])
        # use_gpu = True if (config['drl_agents_parameters']['use_gpu']).lower() == 'true' else False


        dataset = MDPDataset.load(self.dataset_path)
        train_episodes, test_episodes = train_test_split(dataset, test_size=0.3)

        algo = self.algo(n_epochs=nb_epochs, q_func_type="mean", use_gpu=self.use_gpu,  gamma=gamma, n_critics=2, save_metrics=False, verbose=False )
        algo.fit(train_episodes,
                eval_episodes=test_episodes,
                n_epochs=nb_epochs,
                save_metrics=True,
                experiment_name=self.algo_name,
                logdir=self.models_directory,
                with_timestamp=False,
                show_progress=False,
                verbose=False,
                shuffle=False,
                scorers={
                    # 'environment': evaluate_on_environment(self.bt_env_d3, n_trials=1),
                    'environment': evaluate_on_environment(self.train_env, n_trials=1),
                    # 'td_error': td_error_scorer,
                    'td_error': continuous_action_diff_scorer,
                } 
                )

        #####################################################################################################
        ### Backtesting - generate bt csvs and insert desired results in db
        #####################################################################################################
        print('backtesting...')
        # import pathlib
        # for file in pathlib.Path(self.path_for_saving_models).glob("*.pt"):
        #     dst = f"{self.model_name_prefix}_{os.path.basename(file)}"
        #     os.rename(file, os.path.join(os.path.dirname(file), dst) )
        
        
        # bt_records = ft_records = []
        # bt_ledger = ft_ledger = 
        best_model_name = best_model_path = df_best_ledger = None
        tmp_bt_metric = tmp_bt_metric_last = -10000
        list_of_models_path = glob.glob(self.path_for_saving_models +'/*.pt')

        df_input = pd.concat([self.obj_dataset.X_backtest, self.obj_dataset.X_forwardtest])
        df_backtest = pd.concat([self.obj_dataset.df_backtest, self.obj_dataset.df_forwardtest])


        ### loop over all saved models during training
        for i in range(len(list_of_models_path)):
            model_to_load = list_of_models_path[i]
            tmp_model = self.algo.from_json(self.path_for_saving_models +'/params.json')
            tmp_model.load_model(model_to_load) 
            # # model_name = str(os.path.basename(list_of_models_path[i])).split('.pt')[0]
            # model_name = self.algo_name + '_' + str(os.path.basename(list_of_models_path[i])).split('.pt')[0]


            # model_number = str(os.path.basename(list_of_models_path[i])).split('.pt')[0].split('_')[-1]
            # model_name = self.algo_name + '_' + str(os.path.basename(list_of_models_path[i])).split('.pt')[0]

            # # tmp_model_name = self.model_name_prefix.split('_')
            # # tmp_model_name.insert(4, model_number)
            # # tmp_model_name =  "_".join(tmp_model_name)

            ### backtest
            bt_ledger_, bt_ending_balance, bt_sharpe, bt_r2, bt_pnl_percent, df_pred = utils.predict_and_backtest_drl(self.library, \
                                            'backtest', tmp_model, df_input, df_backtest, self.instrument_exchange_timehorizon_utc, self.transaction_fee, self.take_profit_percent, self.stop_loss_percent, self.leverage)
            # bt_record = [model_name, bt_ending_balance, bt_pnl_percent, bt_r2, bt_sharpe]
            bt_record = [self.model_name_prefix, bt_ending_balance, bt_pnl_percent, bt_r2, bt_sharpe, utils.get_ledger_days(bt_ledger_)]
            # print(bt_record)

            pred_long_percent, pred_short_percent, acc_overal, acc_long, acc_short = utils.get_accuracies(df_pred, bt_ledger_, self.transaction_fee)

            if pred_long_percent >= 0.4 and pred_short_percent >= 0.4:
                if acc_long >= 0.5 and acc_short >= 0.5:
                    if bt_sharpe >= 2 and bt_r2 >= 0.85:
                        if bt_sharpe > tmp_bt_metric:
                            tmp_bt_metric = bt_sharpe
                            best_model_name = list_of_models_path[i]
                            best_model_path = list_of_models_path[i]
                            self.best_model = True
                            df_best_ledger = bt_ledger_
            
            # print(not(self.best_model), self.obj_dataset.rolling_window_size, self.rolling_window_length, bt_sharpe, tmp_bt_metric_last)
            # print(not(self.best_model), self.obj_dataset.rolling_window_size, self.rolling_window_length, bt_sharpe, tmp_bt_metric)
            if not(self.best_model):
                if self.obj_dataset.rolling_window_size == self.rolling_window_length:
                    if bt_sharpe > tmp_bt_metric_last:
                        tmp_bt_metric_last = bt_sharpe
                        best_model_name = list_of_models_path[i]
                        best_model_path = list_of_models_path[i]
                        df_best_ledger = bt_ledger_
            
            # if bt_sharpe > tmp_bt_metric:
            #     tmp_bt_metric = bt_sharpe
            #     best_model_name = list_of_models_path[i]
            #     best_model_path = list_of_models_path[i]
            

        if not(best_model_name == None):
            bt_ledger, bt_ending_balance, bt_sharpe, bt_r2, bt_pnl_percent, _ = utils.predict_and_backtest_drl(self.library, \
                                            'backtest', tmp_model, self.obj_dataset.X_backtest, self.obj_dataset.df_backtest, self.instrument_exchange_timehorizon_utc, self.transaction_fee, self.take_profit_percent, self.stop_loss_percent, self.leverage)
            # bt_record = [model_name, bt_ending_balance, bt_pnl_percent, bt_r2, bt_sharpe]
            bt_record = [self.model_name_prefix, bt_ending_balance, bt_pnl_percent, bt_r2, bt_sharpe, utils.get_ledger_days(bt_ledger)]

            ### realworld backtest
            ft_ledger, ft_ending_balance, ft_sharpe, ft_r2, ft_pnl_percent, _ = utils.predict_and_backtest_drl(self.library, \
                                        'forwardtest', tmp_model, self.obj_dataset.X_forwardtest, self.obj_dataset.df_forwardtest, self.instrument_exchange_timehorizon_utc, self.transaction_fee, self.take_profit_percent, self.stop_loss_percent, self.leverage)
            # ft_record = [model_name, ft_ending_balance, ft_pnl_percent, ft_r2, ft_sharpe]
            ft_record = [self.model_name_prefix, ft_ending_balance, ft_pnl_percent, ft_r2, ft_sharpe, utils.get_ledger_days(ft_ledger)]

            # # print('\n', model_number, model_name, tmp_model_name)
            # print('\n', list_of_models_path[i])
            # print(bt_record)
            # print(ft_record)


            # if (bt_ending_balance > -100000) and (ft_ending_balance > -100000):
            #     if (bt_sharpe >= tmp_bt_metric) and (ft_sharpe >= tmp_ft_metric):
            #         # print('selected')
            #         tmp_bt_metric = bt_sharpe
            #         tmp_ft_metric = ft_sharpe
            #         bt_records = bt_record
            #         ft_records = ft_record
            #         bt_ledger = bt_ledger_
            #         ft_ledger = ft_ledger_
            #         best_model_name = list_of_models_path[i]
        
        # ### insert into db if positive           
        # if len(bt_records) > 0:

            print(self.model_name_prefix)      
            model_to_load = best_model_name
            tmp_model = self.algo.from_json(self.path_for_saving_models +'/params.json')
            tmp_model.load_model(model_to_load) 
            new_path = f"{self.models_directory}/{self.model_name_prefix}.pt" 
            tmp_model.save_policy(os.path.join(self.path_for_saving_models, new_path) )

            utils.insert_db_drl(self.path_bt_db, self.table_name, [bt_record])
            utils.insert_db_drl(self.path_ft_db, self.table_name, [ft_record])

            bt_ledger.to_csv(os.path.join(self.path_bt_results, f'bt_ledger_{self.model_name_prefix}.csv'))
            ft_ledger.to_csv(os.path.join(self.path_ft_results, f'ft_ledger_{self.model_name_prefix}.csv'))

            # print(df_best_ledger)
            # print(best_model_name)

            df_best_ledger.set_index('datetime', inplace=True)
            df_best_ledger = df_best_ledger[df_best_ledger['sell_price'] != 0]
            df_best_ledger['pnl'] = df_best_ledger['pnl'] + self.transaction_fee
            df_best_ledger = df_best_ledger['pnl']/100
            df_best_ledger = df_best_ledger.resample('D').sum()
            # First localize to UTC, then convert to None (making it timezone-naive)
            df_best_ledger.index = pd.to_datetime(df_best_ledger.index)
            df_best_ledger.index = df_best_ledger.index.tz_localize('UTC').tz_convert(None)
            path_report = os.path.join(self.models_directory, f'{self.model_name_prefix}.html')
            title_report = str(self.algo).split('.')[-2].upper() + ' Results'
            qs.reports.html(df_best_ledger, title=title_report, output=True, compounded=False, download_filename=path_report)
            # qs.reports.html(df_best_ledger, output=True, compounded=False, download_filename=path_report)

        utils.delete_dir(self.path_for_saving_models)
                    

            # utils.insert_db_drl(self.path_bt_db, self.algo_name, bt_records)
            # utils.insert_db_drl(self.path_rw_db, self.algo_name, rw_records)


        # print('csv generation...')
        # arg_vector = [self.algo, list_of_models_path, self.backtest_env, self.path_bt_models_predictions, self.algo_name]
        # obj_generate_bt_csvs = GenerateBacktestCSVs(arg_vector)
        # obj_generate_bt_csvs.d3(self.path_for_saving_models)

        # ### perform backtest and insert into db 
        # print('backtesting...')
        # arg_vector = [self.algo, self.path_bt_models_predictions, self.algo_name, self.current_dir, self.path_bt_db, self.path_bt_results, list_of_models_path, 'bt', self.instrument_exchange_timehorizon_utc]
        # obj_generate_bt_db = GenerateBacktestDB(arg_vector)
        # run_rw = obj_generate_bt_db.meeting_r2_threshold()

        # #####################################################################################################
        # ### Realworld test - generate rw csvs and insert desired results in db
        # #####################################################################################################
        # if run_rw and self.execute_rw_backtest:
        #     ### rw test csvs generation
        #     print('realworld testing...')
        #     list_of_models_path = glob.glob(self.path_for_saving_models +'/*.pt')
        #     arg_vector = [self.algo, list_of_models_path, self.realworld_env, self.path_rw_models_predictions, self.algo_name]
        #     obj_generate_bt_csvs = GenerateBacktestCSVs(arg_vector)
        #     obj_generate_bt_csvs.d3(self.path_for_saving_models)

        #     ### perform backtest and insert into db
        #     arg_vector = [self.algo, self.path_rw_models_predictions, self.algo_name, self.current_dir, self.path_rw_db, self.path_rw_results, list_of_models_path, 'rw', self.instrument_exchange_timehorizon_utc]
        #     obj_generate_bt_db = GenerateBacktestDB(arg_vector)
        #     obj_generate_bt_db.run()              

class D3Retraining():    
    def __init__(self, arg_vector):
        self.algo = arg_vector[0]
        self.current_dir = arg_vector[1]
        self.train_env = arg_vector[2]
        self.instrument_exchange_timehorizon_utc = arg_vector[3]
        self.library = arg_vector[4]
        self.trained_model = arg_vector[5]
        self.obj_dataset = arg_vector[6]
        self.rolling_window_length = arg_vector[7]
        self.transaction_fee = arg_vector[8]
        self.take_profit_percent = arg_vector[9]
        self.stop_loss_percent = arg_vector[10]
        self.leverage = arg_vector[11]
        self.use_gpu = arg_vector[12]
        self.new_model_name = arg_vector[13]

        ### directory paths for backtest and realworld (results and db)
        self.algo_name = self.library + '_' + str(self.algo).split('.')[-2]

        ### delete model's exisiting data, i.e., results, db, and model predictions
        self.path_bt_results, self.path_bt_db, self.path_ft_results, \
                self.path_ft_db, self.path_for_saving_models  = utils.get_paths_drl(self.current_dir, self.algo_name)

        self.models_directory = os.path.join(self.current_dir, models_directory)

        ### name prefix for models saving
        th = f'{self.instrument_exchange_timehorizon_utc[2]}'
        instrument = f'{self.instrument_exchange_timehorizon_utc[0]}'.upper()
        utc = f'{self.instrument_exchange_timehorizon_utc[3]}'
        ds = f'{self.obj_dataset.train_test_details}'
        algo = f'{self.algo_name}'
        sw = f'{self.obj_dataset.rolling_window_size}'
        # self.model_name_prefix = f'{th}_{instrument}_{utc}_{ds}_{algo}_{sw}'
        # print(self.model_name_prefix)
        # self.model_name_prefix = f'{instrument}_{th}_{algo}_{sw}'
        
        self.model_name_prefix = f'{instrument}_{th}_{algo}_{sw}_{utc}_{ds}'

        self.table_name = f'{instrument}_{th}'
        # print(self.model_name_prefix)      
        # print('td3_plus_bc' in self.model_name_prefix)      
        if 'td3_plus_bc' in self.model_name_prefix:
            self.model_name_prefix = self.model_name_prefix.replace('td3_plus_bc', 'td3plusbc')
        # print(self.model_name_prefix)      
        self.best_model = False

        # ### delete existing data if exists
        # utils.delete_exisiting_data(self.algo_name, self.table_name, self.path_bt_results, self.path_bt_db, self.path_ft_results, self.path_ft_db, self.models_directory)

    def create_dd_dataset(self):
        from d3rlpy.dataset import MDPDataset
        env = self.train_env
        observations = []
        actions = []
        rewards = []
        terminals = []
        for i in range (1000):
            env.reset_global_timestep()
            state = env.reset()
            observations.append(state)
            while True:
                # current_index = env.global_timestep-1
                # next_index = env.global_timestep
                # action = [env.np_labels[current_index][env.labels_target_index]]
                obs = np.array([state], dtype= np.float32)
                obs = torch.from_numpy(obs)
                # print(len(obs), self.trained_model)
                action = self.trained_model(obs)[0].tolist()

                
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
        self.dataset = MDPDataset(observations, actions, rewards, terminals, discrete_action=False)
        self.dataset.episodes
        episode = self.dataset.episodes[0]
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
        # dataset.dump(dataset_path)

    def score_strategy(self, strategy_metrics, weights):
        # Normalize the metrics
        min_max_scaler = MinMaxScaler()
        normalized_metrics = min_max_scaler.fit_transform(strategy_metrics)

        # Calculate the score for each strategy
        scores = np.dot(normalized_metrics, weights)

        # Find the index of the best strategy
        best_strategy_index = np.argmax(scores)

        return best_strategy_index

    def train(self):

        # import logging
        # import warnings
        # logging.getLogger('lightning').setLevel(0)
        # logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
        # warnings.simplefilter(action='ignore', category=FutureWarning)
        # warnings.filterwarnings('ignore')
        # import warnings
        # from torch.serialization import SourceChangeWarning
        # warnings.filterwarnings("ignore", category=SourceChangeWarning)

        #####################################################################################################
        ### training
        #####################################################################################################
        # print('training...')
        if os.path.exists(self.path_for_saving_models) and os.path.isdir(self.path_for_saving_models):
            shutil.rmtree(self.path_for_saving_models)

        self.create_dd_dataset()

        from d3rlpy.dataset import MDPDataset
        from sklearn.model_selection import train_test_split
        from d3rlpy.metrics.scorer import evaluate_on_environment, td_error_scorer, value_estimation_std_scorer, continuous_action_diff_scorer, discounted_sum_of_advantage_scorer, average_value_estimation_scorer, discrete_action_match_scorer
        # nb_epochs = int(config['drl_agents_parameters']['nb_epochs'])
        # use_gpu = True if (config['drl_agents_parameters']['use_gpu']).lower() == 'true' else False


        # nb_epochs = 400

        # dataset = MDPDataset.load(self.dataset_path)
        dataset = self.dataset
        train_episodes, test_episodes = train_test_split(dataset, test_size=0.3)

        algo = self.algo(n_epochs=nb_epochs, q_func_type="mean", use_gpu=self.use_gpu,  gamma=gamma, n_critics=2, save_metrics=False, verbose=False )
        algo.fit(train_episodes,
                eval_episodes=test_episodes,
                n_epochs=nb_epochs,
                save_metrics=True,
                experiment_name=self.algo_name,
                logdir=self.models_directory,
                with_timestamp=False,
                show_progress=False,
                verbose=False,
                shuffle=False,
                scorers={
                    # 'environment': evaluate_on_environment(self.bt_env_d3, n_trials=1),
                    'environment': evaluate_on_environment(self.train_env, n_trials=1),
                    # 'td_error': td_error_scorer,
                    'td_error': continuous_action_diff_scorer,
                } 
                )

        #####################################################################################################
        ### Backtesting - generate bt csvs and insert desired results in db
        #####################################################################################################
        # print('backtesting...')
        # # import pathlib
        # # for file in pathlib.Path(self.path_for_saving_models).glob("*.pt"):
        # #     dst = f"{self.model_name_prefix}_{os.path.basename(file)}"
        # #     os.rename(file, os.path.join(os.path.dirname(file), dst) )
        
        
        # bt_records = ft_records = []
        # bt_ledger = ft_ledger = 
        best_model_name = best_model_path = df_best_ledger = None
        tmp_bt_metric = tmp_bt_metric_last = -10000
        list_of_models_path = glob.glob(self.path_for_saving_models +'/*.pt')

        df_input = pd.concat([self.obj_dataset.X_backtest, self.obj_dataset.X_forwardtest])
        df_backtest = pd.concat([self.obj_dataset.df_backtest, self.obj_dataset.df_forwardtest])

        strategy_metrics = np.empty((0,4), float)

        ### loop over all saved models during training
        for i in range(len(list_of_models_path)):
            model_to_load = list_of_models_path[i]
            tmp_model = self.algo.from_json(self.path_for_saving_models +'/params.json')
            tmp_model.load_model(model_to_load) 
            # # model_name = str(os.path.basename(list_of_models_path[i])).split('.pt')[0]
            # model_name = self.algo_name + '_' + str(os.path.basename(list_of_models_path[i])).split('.pt')[0]


            # model_number = str(os.path.basename(list_of_models_path[i])).split('.pt')[0].split('_')[-1]
            # model_name = self.algo_name + '_' + str(os.path.basename(list_of_models_path[i])).split('.pt')[0]

            # # tmp_model_name = self.model_name_prefix.split('_')
            # # tmp_model_name.insert(4, model_number)
            # # tmp_model_name =  "_".join(tmp_model_name)

            ### backtest
            bt_ledger_, bt_ending_balance, bt_sharpe, bt_r2, bt_pnl_percent, df_pred = utils.predict_and_backtest_drl(self.library, \
                                            'backtest', tmp_model, df_input, df_backtest, self.instrument_exchange_timehorizon_utc, self.transaction_fee, self.take_profit_percent, self.stop_loss_percent, self.leverage)
            bt_record = [i, bt_ending_balance, bt_pnl_percent, bt_r2, bt_sharpe]
            # bt_record = [self.model_name_prefix, bt_ending_balance, bt_pnl_percent, bt_r2, bt_sharpe, utils.get_ledger_days(bt_ledger_)]
            # print(model_to_load)
            # print(bt_record)
            l = [bt_ending_balance, bt_sharpe, bt_r2, bt_pnl_percent]

            strategy_metrics = np.vstack([strategy_metrics,l])

        # print(strategy_metrics)
        weights = np.array([0.25, 0.25, 0.25, 0.25])

        best_strategy_index = self.score_strategy(strategy_metrics, weights)
        best_model_name = str(os.path.basename(list_of_models_path[best_strategy_index]))
        # print(f"The best strategy is Strategy {best_strategy_index}", list_of_models_path[best_strategy_index], best_model_name)
        # best_model_path

        model_to_load = list_of_models_path[best_strategy_index]
        tmp_model = self.algo.from_json(self.path_for_saving_models +'/params.json')
        tmp_model.load_model(model_to_load) 
        # new_path = f"{self.models_directory}/{self.model_name_prefix}.pt" 
        new_path = f"{self.models_directory}/{self.new_model_name}.pt" 
        tmp_model.save_policy(os.path.join(self.path_for_saving_models, new_path) )


        model_to_load = list_of_models_path[best_strategy_index]
        tmp_model = self.algo.from_json(self.path_for_saving_models +'/params.json')
        tmp_model.load_model(model_to_load) 

        # bt_ledger_, bt_ending_balance, bt_sharpe, bt_r2, bt_pnl_percent, df_pred = utils.predict_and_backtest_drl(self.library, \
        #                                     'backtest', tmp_model, df_input, df_backtest, self.instrument_exchange_timehorizon_utc, self.transaction_fee, self.take_profit_percent, self.stop_loss_percent, self.leverage)
        # bt_record = [i, bt_ending_balance, bt_pnl_percent, bt_r2, bt_sharpe]
        # # bt_record = [self.model_name_prefix, bt_ending_balance, bt_pnl_percent, bt_r2, bt_sharpe, utils.get_ledger_days(bt_ledger_)]
        # # print(bt_record)

        # bt_ledger_.to_csv(os.path.join(self.current_dir, f'bt_ledger_{self.model_name_prefix}.csv'))
        # df_best_ledger = bt_ledger_
        # df_best_ledger.set_index('datetime', inplace=True)
        # df_best_ledger = df_best_ledger[df_best_ledger['sell_price'] != 0]
        # df_best_ledger['pnl'] = df_best_ledger['pnl'] + self.transaction_fee
        # df_best_ledger = df_best_ledger['pnl']/100
        # df_best_ledger.index = df_best_ledger.index.tz_convert(None)
        # # path_report = os.path.join(self.models_directory, f'{best_model_name}.html')
        # path_report = os.path.join(self.current_dir, f'bt_ledger_{self.model_name_prefix}.html')
        # title_report = str(self.algo).split('.')[-2].upper() + ' Results'
        # qs.reports.html(df_best_ledger, title=title_report, output=True, compounded=False, download_filename=path_report)


        utils.delete_dir(self.path_for_saving_models)         
	
class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions

    def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

    def __len__(self):
        return len(self.observations)

class SBPretraining():
    def __init__(self, arg_vector):
        self.trained_model = arg_vector[0]
        self.train_env = arg_vector[1]
        self.obj_dataset = arg_vector[2]
        self.current_dir = arg_vector[3]
        self.new_model = arg_vector[4]

        # df_input = pd.concat([self.obj_dataset.X_backtest, self.obj_dataset.X_forwardtest])
        # df_backtest = pd.concat([self.obj_dataset.df_backtest, self.obj_dataset.df_forwardtest])

        # instrument_exchange_timehorizon_utc = ['btc', 'dydx', '24h', 0]
        # # bt_ledger_, bt_ending_balance, bt_sharpe, bt_r2, bt_pnl_percent, df_pred = utils.predict_and_backtest_drl('sb3', \
        # #                                     'backtest', self.trained_model, df_input, df_backtest, instrument_exchange_timehorizon_utc, 0.05, 100, 4, 1)
        # # print(bt_ending_balance, bt_sharpe, bt_r2, bt_pnl_percent)

        # # bt_ledger_, bt_ending_balance, bt_sharpe, bt_r2, bt_pnl_percent, df_pred = utils.predict_and_backtest_drl('sb3', \
        # #                                     'backtest', self.new_model, df_input, df_backtest, instrument_exchange_timehorizon_utc, 0.05, 100, 4, 1)
        # # print(bt_ending_balance, bt_sharpe, bt_r2, bt_pnl_percent)

        # # print(self.train_env.training_length)


        self.create_expert_dataset()

        self.pretrain_agent(
            epochs=30,
            scheduler_gamma=0.7,
            learning_rate=1.0,
            log_interval=100,
            no_cuda=True,
            seed=1,
            batch_size=64,
            test_batch_size=1000,
        )

        # bt_ledger_, bt_ending_balance, bt_sharpe, bt_r2, bt_pnl_percent, df_pred = utils.predict_and_backtest_drl('sb3', \
        #                                     'backtest', self.new_model, df_input, df_backtest, instrument_exchange_timehorizon_utc, 0.05, 100, 4, 1)
        # print(bt_ending_balance, bt_sharpe, bt_r2, bt_pnl_percent)

           
    def create_expert_dataset(self):

            
        num_interactions = int(self.train_env.training_length)
        # num_interactions = int(1e3)

        if isinstance(self.train_env.action_space, gym.spaces.Box):
            expert_observations = np.empty((num_interactions,) + self.train_env.observation_space.shape)
            expert_actions = np.empty((num_interactions,) + (self.train_env.action_space.shape[0],))

        else:
            expert_observations = np.empty((num_interactions,) + self.train_env.observation_space.shape)
            expert_actions = np.empty((num_interactions,) + self.train_env.action_space.shape)

        obs = self.train_env.reset()

        for i in tqdm(range(num_interactions)):
            action, _ = self.trained_model.predict(obs, deterministic=True)
            expert_observations[i] = obs
            expert_actions[i] = action
            obs, reward, terminated, info = self.train_env.step(action)
            done = terminated
            if done:
                obs = self.train_env.reset()

        # expert_dataset_path = os.path.join(current_dir, 'expert_data.npz')


        expert_dataset = ExpertDataSet(expert_observations, expert_actions)

        train_size = int(0.8 * len(expert_dataset))

        test_size = len(expert_dataset) - train_size

        self.train_expert_dataset, self.test_expert_dataset = random_split(
            expert_dataset, [train_size, test_size]
        )

        # print("test_expert_dataset: ", len(self.test_expert_dataset))
        # print("train_expert_dataset: ", len(self.train_expert_dataset))


    
    def pretrain_agent(self, 
        # student,
        batch_size=64,
        epochs=1000,
        scheduler_gamma=0.7,
        learning_rate=1.0,
        log_interval=100,
        no_cuda=True,
        seed=1,
        test_batch_size=64,
    ):
        use_cuda = not no_cuda and th.cuda.is_available()
        th.manual_seed(seed)
        device = th.device("cuda" if use_cuda else "cpu")
        kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

        if isinstance(self.train_env.action_space, gym.spaces.Box):
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        # Extract initial policy
        model = self.new_model.policy.to(device)

        def train(model, device, train_loader, optimizer):
            model.train()

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                if isinstance(self.train_env.action_space, gym.spaces.Box):
                    # A2C/PPO policy outputs actions, values, log_prob
                    # SAC/TD3 policy outputs actions only
                    if isinstance(self.new_model, (A2C, PPO)):
                        action, _, _ = model(data)
                    else:
                        # SAC/TD3:
                        action = model(data)
                    action_prediction = action.double()
                else:
                    # Retrieve the logits for A2C/PPO when using discrete actions
                    dist = model.get_distribution(data)
                    action_prediction = dist.distribution.logits
                    target = target.long()

                loss = criterion(action_prediction, target)
                loss.backward()
                optimizer.step()
                # if batch_idx % log_interval == 0:
                #     print(
                #         "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #             epoch,
                #             batch_idx * len(data),
                #             len(train_loader.dataset),
                #             100.0 * batch_idx / len(train_loader),
                #             loss.item(),
                #         )
                #     )

        def test(model, device, test_loader):
            model.eval()
            test_loss = 0
            with th.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)

                    if isinstance(self.train_env.action_space, gym.spaces.Box):
                        # A2C/PPO policy outputs actions, values, log_prob
                        # SAC/TD3 policy outputs actions only
                        if isinstance(self.new_model, (A2C, PPO)):
                            action, _, _ = model(data)
                        else:
                            # SAC/TD3:
                            action = model(data)
                        action_prediction = action.double()
                    else:
                        # Retrieve the logits for A2C/PPO when using discrete actions
                        dist = model.get_distribution(data)
                        action_prediction = dist.distribution.logits
                        target = target.long()

                    test_loss = criterion(action_prediction, target)
            test_loss /= len(test_loader.dataset)
            # print(f"Test set: Average loss: {test_loss:.4f}")

        # Here, we use PyTorch `DataLoader` to our load previously created `ExpertDataset` for training
        # and testing
        train_loader = th.utils.data.DataLoader(
            dataset=self.train_expert_dataset, batch_size=batch_size, shuffle=True, **kwargs
        )
        test_loader = th.utils.data.DataLoader(
            dataset=self.test_expert_dataset,
            batch_size=test_batch_size,
            shuffle=True,
            **kwargs,
        )

        # Define an Optimizer and a learning rate schedule.
        optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

        # Now we are finally ready to train the policy model.
        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer)
            test(model, device, test_loader)
            scheduler.step()

        # Implant the trained policy network back into the RL student agent
        self.new_model.policy = model

class SBRetraining():
    def __init__(self, arg_vector):
        self.algo = arg_vector[0]
        self.current_dir = arg_vector[1]
        self.train_env = arg_vector[2]
        # self.backtest_env = arg_vector[3]
        # self.realworld_env = arg_vector[4]
        # self.execute_rw_backtest = arg_vector[5]
        self.instrument_exchange_timehorizon_utc = arg_vector[3]
        self.library = arg_vector[4]
        self.obj_dataset = arg_vector[5]
        self.rolling_window_length = arg_vector[6]
        self.transaction_fee = arg_vector[7]
        self.take_profit_percent = arg_vector[8]
        self.stop_loss_percent = arg_vector[9]
        self.leverage = arg_vector[10]
        self.use_gpu = arg_vector[11]
        self.new_model = arg_vector[12]
        self.new_model_name = arg_vector[13]

        # arg_vector = [sb_mf_algo_array[i], current_dir, train_env, instrument_exchange_timehorizon_utc, library, obj_dataset]

        ### directory paths for backtest and realworld (results and db)
        self.algo_name = self.library + '_' + str(self.algo).split('.')[-2]

        ### delete model's exisiting data, i.e., results, db, and model predictions
        self.path_bt_results, self.path_bt_db, self.path_ft_results, \
                self.path_ft_db, self.path_for_saving_models  = utils.get_paths_drl(self.current_dir, self.algo_name)

        self.models_directory = os.path.join(self.current_dir, models_directory)

        th = f'{self.instrument_exchange_timehorizon_utc[2]}'
        instrument = f'{self.instrument_exchange_timehorizon_utc[0]}'.upper()
        utc = f'{self.instrument_exchange_timehorizon_utc[3]}'
        ds = f'{self.obj_dataset.train_test_details}'
        algo = f'{self.algo_name}'
        sw = f'{self.obj_dataset.rolling_window_size}'
        # self.model_name_prefix = f'{instrument}_{th}_{algo}_{sw}'
        # print(self.model_name_prefix)
        self.model_name_prefix = f'{instrument}_{th}_{algo}_{sw}_{utc}_{ds}'
        # self.model_name_prefix = self.new_model_name
        # self.model_name_prefix = f'{th}_{instrument}_{utc}_{ds}_{algo}_{sw}'
        # print(self.model_name_prefix)
        self.table_name = f'{instrument}_{th}'
        # print(self.model_name_prefix)       
        self.best_model = False

        # ### delete existing data if exists
        # utils.delete_exisiting_data(self.algo_name, self.table_name, self.path_bt_results, self.path_bt_db, self.path_ft_results, self.path_ft_db, self.models_directory)
                
    def train(self):
        #####################################################################################################
        ### training
        #####################################################################################################
        print('Retraining...')
        if os.path.exists(self.path_for_saving_models) and os.path.isdir(self.path_for_saving_models):
            shutil.rmtree(self.path_for_saving_models)

        device = 'cpu'
        if self.use_gpu :
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.library == 'sb3':
            from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
            # model = self.algo("MlpPolicy", self.train_env, verbose=0, gamma=gamma, device=device)

        # total_timesteps = 400000
        # save_frequency = 1000

        checkpoint_callback = CheckpointCallback(save_freq=save_frequency, save_path=self.path_for_saving_models, name_prefix=self.model_name_prefix)
        # model.learn(total_timesteps=total_timesteps,callback=checkpoint_callback)
        self.new_model.learn(total_timesteps=total_timesteps,callback=checkpoint_callback)


        #####################################################################################################
        ### Backtesting
        #####################################################################################################
        # print('backtesting...')
        # bt_records = ft_records = []
        # bt_ledger = ft_ledger = 
        best_model_name = best_model_path = df_best_ledger = None
        tmp_bt_metric = tmp_bt_metric_last = -10000
        list_of_models_path = glob.glob(self.path_for_saving_models +'/*.zip')


        df_input = pd.concat([self.obj_dataset.X_backtest, self.obj_dataset.X_forwardtest])
        df_backtest = pd.concat([self.obj_dataset.df_backtest, self.obj_dataset.df_forwardtest])

        # print(df_input)
        # print(df_backtest)

        
        strategy_metrics = np.empty((0,4), float)

        ### loop over all saved models during training
        for i in range(len(list_of_models_path)):
            tmp_model = self.algo.load(list_of_models_path[i])
            model_name = str(os.path.basename(list_of_models_path[i])).split('.zip')[0]
            # print(model_name)
            # model_name_shortened = model_name.split('_steps')[0]
            

            ### backtest
            bt_ledger_, bt_ending_balance, bt_sharpe, bt_r2, bt_pnl_percent, df_pred = utils.predict_and_backtest_drl(self.library, \
                                            'backtest', tmp_model, df_input, df_backtest, self.instrument_exchange_timehorizon_utc, self.transaction_fee, self.take_profit_percent, self.stop_loss_percent, self.leverage)
            
            # print(i, bt_ending_balance, bt_sharpe, bt_r2, bt_pnl_percent)

            l = [bt_ending_balance, bt_sharpe, bt_r2, bt_pnl_percent]

            strategy_metrics = np.vstack([strategy_metrics,l])

        # print(strategy_metrics)
        weights = np.array([0.25, 0.25, 0.25, 0.25])

        best_strategy_index = self.score_strategy(strategy_metrics, weights)
        best_model_name = str(os.path.basename(list_of_models_path[best_strategy_index]))
        # print(f"The best strategy is Strategy {best_strategy_index}", list_of_models_path[best_strategy_index], best_model_name)


        # model_to_copy = os.path.join(self.path_for_saving_models, f'{best_model_name}.zip')
        model_to_copy = list_of_models_path[best_strategy_index]
        shutil.copy(model_to_copy, self.models_directory)

        model_to_rename = os.path.join(self.models_directory, f'{best_model_name}.zip')
        # best_model_name = best_model_name.split('_')[:-2] 
        # best_model_name = "_".join(best_model_name)
        new_path = f"{self.models_directory}/{self.new_model_name}.zip"        
        shutil.move(model_to_rename, new_path)

        # tmp_model = self.algo.load(list_of_models_path[best_strategy_index])
        # bt_ledger_, bt_ending_balance, bt_sharpe, bt_r2, bt_pnl_percent, df_pred = utils.predict_and_backtest_drl(self.library, \
        #                                     'backtest', tmp_model, df_input, df_backtest, self.instrument_exchange_timehorizon_utc, self.transaction_fee, self.take_profit_percent, self.stop_loss_percent, self.leverage)
            

        # bt_ledger_.to_csv(os.path.join(self.current_dir, f'bt_ledger_{self.model_name_prefix}.csv'))
        # df_best_ledger = bt_ledger_
        # df_best_ledger.set_index('datetime', inplace=True)
        # df_best_ledger = df_best_ledger[df_best_ledger['sell_price'] != 0]
        # df_best_ledger['pnl'] = df_best_ledger['pnl'] + self.transaction_fee
        # df_best_ledger = df_best_ledger['pnl']/100
        # df_best_ledger.index = df_best_ledger.index.tz_convert(None)
        # # path_report = os.path.join(self.models_directory, f'{best_model_name}.html')
        # path_report = os.path.join(self.current_dir, f'bt_ledger_{self.model_name_prefix}.html')
        # title_report = str(self.algo).split('.')[-2].upper() + ' Results'
        # qs.reports.html(df_best_ledger, title=title_report, output=True, compounded=False, download_filename=path_report)

        utils.delete_dir(self.path_for_saving_models)


    def score_strategy(self, strategy_metrics, weights):
        # Normalize the metrics
        min_max_scaler = MinMaxScaler()
        normalized_metrics = min_max_scaler.fit_transform(strategy_metrics)

        # Calculate the score for each strategy
        scores = np.dot(normalized_metrics, weights)

        # Find the index of the best strategy
        best_strategy_index = np.argmax(scores)

        return best_strategy_index