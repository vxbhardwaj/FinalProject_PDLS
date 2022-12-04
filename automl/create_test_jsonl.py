# Run as python create_test_jsonl.py test_data_path gcs_path

import csv
import pandas as pd
import math
import defopt
from PIL import Image
import numpy as np
import os

def create_jsonl(test_data_path: str, gcs_path: str):
    all_data = []
    for file in os.listdir(test_data_path):
        new_row = {"content": gcs_path + "/" + file + ".png", "mimeType": "image/png"}
        all_data.append(str(new_row))

    with open("automl_test_json.jsonl", "w") as f:
        f.write("\n".join(all_data))

if __name__ == '__main__':
    defopt.run(create_jsonl)