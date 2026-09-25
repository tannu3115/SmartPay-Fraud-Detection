@echo off
TITLE SmartPay Guard - ML Fraud Detection
echo ========================================================
echo        SmartPay Guard - ML Fraud Detection System
echo ========================================================
echo.
echo Step 1: Checking Python installation...
python --version
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH! Please install Python.
    pause
    exit /b
)

echo.
echo Step 2: Training model and checking dataset...
python models/train_model.py
if errorlevel 1 (
    echo [ERROR] Failed to train model.
    pause
    exit /b
)

echo.
echo Step 3: Launching Streamlit Web Application...
echo The app will open in your default web browser at http://localhost:8501
echo To stop the application at any time, press Ctrl+C in this terminal window.
echo.
python -m streamlit run app.py
pause
