@echo off
echo Starting PARMS Jupyter Notebook Analysis...
echo.
echo Activating virtual environment and launching Jupyter...
echo.

REM Activate virtual environment
call "%~dp0parms_env\Scripts\activate.bat"

REM Launch Jupyter notebook with the virtual environment
"%~dp0parms_env\Scripts\jupyter.exe" notebook PARMS_Analysis.ipynb