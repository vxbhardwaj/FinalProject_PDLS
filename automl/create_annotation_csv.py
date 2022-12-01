# Run as python create_annotation_csv.py csv_path gcs_path

import csv
import pandas as pd
import math
import defopt
from PIL import Image
import numpy as np

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def read_from_csv(csv_path: str, gcs_path: str):
    df = pd.read_csv(csv_path)
    all_data = []
    image_width = 1024
    for index, row in df.iterrows():
        filename = row.get("patientId") + ".dcm.png"
        label = row.get("Target")
        width = row.get("width")
        height = row.get("height")
        xmin = row.get("x")
        ymin = row.get("y")
        gcs_uri = "gs://" + gcs_path + "/" + filename
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
            new_xmin = round((xmin / image_width), 1)
            new_xmax = round((xmax / image_width), 1)
            new_ymin = round((ymin / image_width), 1)
            new_ymax = round((ymax / image_width), 1)
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