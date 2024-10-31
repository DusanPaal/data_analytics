@echo off

REM Activate the pyhton virtual environment
call %~dp0\env\Scripts\activate

REM Run the Jupyter Lab application
jupyter lab --notebook-dir=E:\data_analytics