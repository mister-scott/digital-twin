REM remove old venv instance
REM mv build oldbuild && start /b ( rmdir /S /Q oldbuild & exit )
rmdir /S /Q build

REM remove old venv instance
REM mv dist olddist && start /b ( rmdir /S /Q olddist & exit )
rmdir /S /Q dist

REM Activate venv
call venv\Scripts\activate.bat

REM Build project
pyinstaller .\template.spec