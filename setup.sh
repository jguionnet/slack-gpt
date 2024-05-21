##/bin/bash

source .venv/bin/activate

pip3 install --upgrade pip
pip3 cache purge
python3 -m pip install -r requirements.txt 

source key.txt
