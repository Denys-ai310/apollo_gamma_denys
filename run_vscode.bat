set CONDAPATH=D:\Anaconda
set ENVNAME=apollo_gamma

if %ENVNAME%==base (set ENVPATH=%CONDAPATH%) else (set ENVPATH=%CONDAPATH%\envs\%ENVNAME%)

call %CONDAPATH%\Scripts\activate.bat %ENVPATH%

cd E:\apollo_gamma
code .

