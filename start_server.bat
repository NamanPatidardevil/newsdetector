@echo off
title Veritas Core - Fake News Detection Server
echo ====================================================================
echo Starting Veritas Core AI Detection Engine on Port 8000...
echo ====================================================================
cd /d "%~dp0"
python app.py
pause
