cd "$(dirname "$0")"
nohup bash a2c.sh &    # Run a2c.sh in the background
nohup bash ppo.sh &    # Run ppo.sh in the background
nohup bash ddpg.sh &   # Run ddpg.sh in the background
nohup bash sac.sh &    # Run sac.sh in the background
nohup bash td3.sh &    # Run td3.sh in the background

nohup bash bear.sh &   # Run bear.sh in the background
nohup bash awac.sh &   # Run awac.sh in the background
nohup bash td3plusbc.sh &   # Run td3plusbc.sh in the background
nohup bash cql.sh &    # Run cql.sh in the background
nohup bash plas.sh &   # Run plas.sh in the background

wait