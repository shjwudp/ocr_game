import csv
import os
import json
import wget
from pathlib import Path
import requests

for (root, dirs, files) in os.walk("./index", topdown=True):
    for filename in files:
        if not filename.endswith(".csv"):
            continue

        desc = filename[:-len(".csv")]
        img_dir = os.path.join("./img", desc)
        Path(img_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(root, filename)) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                img_url = json.loads(row["原始数据"])["tfspath"]
                # print("img_url={}".format(img_url))
                # wget.download(img_url, out=img_dir)

                img_name = img_url.split('/')[-1]
                img_data = requests.get(img_url).content
                with open(os.path.join(img_dir, img_name), 'wb') as handler:
                    handler.write(img_data)
