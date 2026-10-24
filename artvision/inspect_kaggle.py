import os
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

files = api.dataset_list_files('ravidussilva/real-ai-art').files
dirs = set()
for f in files:
    parts = f.name.split('/')
    if len(parts) >= 3:
        dirs.add('/'.join(parts[:3]))

for d in sorted(dirs):
    print(d)
