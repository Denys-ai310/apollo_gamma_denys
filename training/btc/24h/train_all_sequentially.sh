cd "$(dirname "$0")"

sh ./generic.sh model_free.py a2c
sh ./generic.sh model_free.py ppo
sh ./generic.sh model_free.py ddpg
sh ./generic.sh model_free.py sac
sh ./generic.sh model_free.py td3

sh ./generic.sh model_based.py bear
sh ./generic.sh model_based.py awac
sh ./generic.sh model_based.py td3plusbc
sh ./generic.sh model_based.py cql
sh ./generic.sh model_based.py plas
