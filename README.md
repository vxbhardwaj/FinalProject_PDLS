# Pneumonia Detection using Deep Learning

## Overview

- In this project, we are predicting pneumonia using chest X-RAY images by predicting bounding boxes around areas of the lungs. 
- We have done a comparative analysis of various Image object detection models on the basis of their mAP (mean average precision) scores, associated costs involved, and the amount of training times.
- Our objective is to experiment with AutoML and different pretrained models (Masked-RCNN and YOLO on V100 and T4 GPUs) to find the best performing one which is cost-effective and takes less human effort to build. 

## Repository Code Structure
Jupyter notebooks can be run as is, with relevant files provided along side.

### automl folder
Files:
- create_annotation_csv.py converts the training annotations provided by Kaggle to format needed by AutoML
```bash
      python create_annotation_csv.py results/automl_annotations.csv gs://sample_bucket/train_images/
```
- create_test_jsonl.py read all test images and stores it in JSONL format with mimeType
```bash
      python create_test_jsonl.py results/automl_test_json.jsonl gs://sample_bucket/test_images/
```
- create_submission_csv.py AutoML run prediction in batches and create a JSONL file for each batch. This script merges all files into the CSV format
required by Kaggle for submission
```bash
      python create_submission_csv.py results/predictions/faster_prediction_model
```
- results: This folder contains AutoML predictions and generate annotation CSV and jsonl files

### dicom_to_png folder contains
This folder contains files related to converting DCM X ray images to PNG format (required for autoML training).

### masked-rcnn folder
This folder contains code used to train Masked R-CNN There are three subfolder
- best_model: This folder contains the notebook and test predictions CSV from Masked R-CNN model that gave the best test mAP scores
- T4-exp: This folder contains the jupyter notebook which show the training and inference time on T4 GPU. t4-loss-acc-summary.csv captures the train/val loss, training time per epoch. t4-loss-plot.png is a plot of the train vs val loss observed.
- V100-exp: This folder follows same structure as NVIDIA-T4-exp but contains results from experiment run on V100 GPU

### yolo folder
This folder contains YOLO v3 related files and has the following two subfolders:
- T4: This folder contains the notebook of YOLO v3 run on T4 GPU that has all train logs and time taken, Loss vs iterations graph, supplemental files, along with configuration files.
- V100: This folder contains the notebook run on V100 GPU containing training and inference logs, time taken, Loss vs iterations graph, supplemental files, along with configuration files. 
##### YOLO v3 on V100 gave the BEST mAP SCORE out of all the scenarios we tested.

## Model weights
Final model weights used to get mAP score on the test set are as follows. We don't have access to weight learned by AutoML. We directly get the test prediction use the batch prediction service.

YOLOv3 weights:

T4 exp -https://storage.googleapis.com/projsub/T4/rsna_yolov3_final.weights <br>
V100 exp - https://storage.googleapis.com/projsub/V100/rsna_yolov3_final.weights

Masked R-CNN weights:

Best model - https://storage.googleapis.com/projsub/MRCNN/best_model/mask_rcnn_pneumonia_final_weights.h5 <br>
T4 exp - https://storage.googleapis.com/projsub/MRCNN/T4/mask_rcnn_final_weights.h5<br>
V100 exp - https://storage.googleapis.com/projsub/MRCNN/V100/mask_rcnn_final_weights.h5


## Results (including charts/tables) and observations:


![image](https://user-images.githubusercontent.com/76259177/208492672-7f0d4dd5-7055-4d4f-87c6-fb806603a5f8.png)

![image](https://user-images.githubusercontent.com/76259177/208493396-864dad8c-814c-4908-b8dd-56088e45a5c2.png)

![image](https://user-images.githubusercontent.com/76259177/208493732-b3ab1f59-bf19-430c-884d-ecd6b3c2ac86.png)

![image](https://user-images.githubusercontent.com/76259177/208494076-806f8c94-90ed-47e0-bd2a-dc87ded58484.png)

![image](https://user-images.githubusercontent.com/76259177/208494261-398fea5c-84f8-48b5-9293-fcb0b9e3879c.png)


## Results Summary

- YOLOv3 model gives the best test mAP score among all our experiments
- Masked RCNN performs inference 22X much faster than YOLO, 5X faster than AutoML and achieves a score only slightly lower than the YOLO, but the training results vary significantly between runs
- YOLOv3 and Masked RCNN did 4.53% better than the best performing AutoML model
- The best performance on the leaderboard for this Kaggle competition was 0.25
- Factoring in human effort and compute cost involved with running masked-RCNN and YOLO v3, AutoML helps build models faster and provides a reasonable benchmark but there is definitely scope for improvement




