REM remove old venv instance
REM mv venv oldvenv && start /b rmdir /S /Q oldvenv
rmdir /S /Q venv

REM create new venv instance
python -m venv venv

REM run installation of dependencies
call install_requirements.bat