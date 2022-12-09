import csv
import defopt
import os
import json

def get_patient_id(path):
    return path.split("/")[-1].split(".")[0]

def jsonl_to_csv(predictions_folder: str):
    all_data = []
    image_width = 1024
    for file in os.listdir(predictions_folder):
        f = open(os.path.join(predictions_folder, file))
        for line in f.readlines():
            j = json.loads(line)
            patient_id = get_patient_id(j["instance"]["content"])
            print(patient_id)
            prediction = j["prediction"]
            prediction_string = ""
            for conf, bbox in zip(prediction["confidences"], prediction["bboxes"]):
                if conf < 0.3:
                    continue
                xmin, xmax, ymin, ymax = [c * image_width for c in bbox]
                height = ymax - ymin
                width = xmax - xmin
                prediction_string += str(conf) + " " + str(xmin) + " " + str(ymin) + " " + str(width) + " " + str(height) + " "
            prediction_string = prediction_string.strip()
            all_data.append([patient_id, prediction_string])

    with open("automl_" + predictions_folder.split('/')[-1] + "_submission.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["patientId", "PredictionString"])
        writer.writerows(all_data)

if __name__ == '__main__':
    defopt.run(jsonl_to_csv)