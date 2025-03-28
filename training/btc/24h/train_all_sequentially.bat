cd /d "%~dp0" 

call generic.bat model_free.py a2c
call generic.bat model_free.py ppo
call generic.bat model_free.py ddpg
call generic.bat model_free.py sac
call generic.bat model_free.py td3

call generic.bat model_based.py bear
call generic.bat model_based.py awac
call generic.bat model_based.py td3plusbc
call generic.bat model_based.py cql
call generic.bat model_based.py plas


