@echo off
cd /d C:\Users\User\hermes-sales-agent
:loop
echo [hermes] Starting bot...
C:\Users\User\AppData\Local\Python\pythoncore-3.14-64\python.exe main.py
echo [hermes] Bot stopped. Restarting in 5 seconds...
timeout /t 5 /nobreak
goto loop
