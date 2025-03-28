cd "$(dirname "$0")"

./generic.sh model_free.py a2c
./generic.sh model_free.py ppo
./generic.sh model_free.py ddpg
./generic.sh model_free.py sac
./generic.sh model_free.py td3

./generic.sh model_based.py bear
./generic.sh model_based.py awac
./generic.sh model_based.py td3plusbc
./generic.sh model_based.py cql
./generic.sh model_based.py plas
