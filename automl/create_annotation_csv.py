# Run as python create_annotation_csv.py csv_path bucket_name

import csv
import pandas as pd
import numpy as np
import math
import defopt

def read_from_csv(csv_path: str, bucket_name: str):
    df = pd.read_csv(csv_path)
    all_data = []
    for index, row in df.iterrows():
        filename = row.get("patientId") + ".jpg"
        label = row.get("Target")
        width = row.get("width")
        height = row.get("height")
        xmin = row.get("x")
        ymin = row.get("y")
        gcs_uri = "gs://" + bucket_name + "/" + filename
        if math.isnan(xmin) and math.isnan(ymin):
            new_row = [
                "UNASSIGNED",
                gcs_uri
                # TODO: Should label be added here?
            ]
        else:
            xmax = xmin + width
            ymax = ymin + height

            # Normalizing co-ordinates
            new_xmin = round((xmin / width), 1)
            new_xmax = round((xmax / width), 1)
            new_ymin = round((ymin / height), 1)
            new_ymax = round((ymax / height), 1)
            new_row = [
                "UNASSIGNED",
                gcs_uri,
                label,
                new_xmin,
                new_ymin,
                "",
                "",
                new_xmax,
                new_ymax,
                "",
                "",
            ]
        all_data.append(new_row)

    with open("automl_annotations.csv", "w") as f:
        write = csv.writer(f)
        write.writerows(all_data)

if __name__ == '__main__':
    defopt.run(read_from_csv)