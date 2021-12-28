call venv\Scripts\activate.bat

python -m pip install --upgrade pip || python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org  --upgrade pip
python -m pip install -r requirements.txt || python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt